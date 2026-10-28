import React, { useState, useRef, useEffect, useMemo } from 'react';
import {
  Typography, Box, Button, Card, CardContent, Grid,
  MenuItem, Select, LinearProgress, Paper, Table, TableBody, 
  TableCell, TableContainer, TableHead, TableRow, Slider, 
  ThemeProvider, createTheme, CssBaseline, IconButton, Divider,
  InputLabel, FormControl, TextField
} from '@mui/material';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import DownloadIcon from '@mui/icons-material/Download';
import TerminalIcon from '@mui/icons-material/Terminal';
import TimelineIcon from '@mui/icons-material/Timeline';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import AssessmentIcon from '@mui/icons-material/Assessment';
import LightModeIcon from '@mui/icons-material/LightMode';
import DarkModeIcon from '@mui/icons-material/DarkMode';
import ImageIcon from '@mui/icons-material/Image';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, 
  Legend, ResponsiveContainer, ReferenceLine, BarChart, Bar, Cell
} from 'recharts';

export default function FairnessApp() {
  const [mode, setMode] = useState('light');

  const [fileId, setFileId] = useState('');
  const [columns, setColumns] = useState([]);
  const [targetCol, setTargetCol] = useState('');
  const [sa1Col, setSa1Col] = useState('');
  const [sa2Col, setSa2Col] = useState('');
  const [modelType, setModelType] = useState('logistic_regression');
  
  const [popSize, setPopSize] = useState(150);
  const [generations, setGenerations] = useState(40);
  
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [eta, setEta] = useState('--');
  const [logs, setLogs] = useState([]);
  const [exportId, setExportId] = useState(null);

  const [baselineMetrics, setBaselineMetrics] = useState(null);
  const [optimizedMetrics, setOptimizedMetrics] = useState(null);
  const [liveGraphData, setLiveGraphData] = useState([]);
  const [finalComparisonData, setFinalComparisonData] = useState([]);

  // Explainer States
  const [explainIndex, setExplainIndex] = useState(0);
  const [explainType, setExplainType] = useState('lime');
  const [explainData, setExplainData] = useState([]);
  const [isExplaining, setIsExplaining] = useState(false);

  const consoleEndRef = useRef(null);

  const theme = useMemo(() => createTheme({
    palette: {
      mode,
      primary: { main: mode === 'dark' ? '#4db8ff' : '#0066cc' },
      success: { main: mode === 'dark' ? '#00ff00' : '#2e7d32' },
      warning: { main: '#ffb84d' },
      error: { main: '#ff5555' },
      background: { 
        default: mode === 'dark' ? '#0d1117' : '#f4f6f8', 
        paper: mode === 'dark' ? '#161b22' : '#ffffff' 
      },
    },
    typography: {
      fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
      h4: { fontWeight: 700, letterSpacing: '-0.02em' },
      h5: { fontWeight: 600 },
      h6: { fontWeight: 600 },
    },
    components: {
      MuiCard: { 
        styleOverrides: { 
          root: { 
            border: mode === 'dark' ? '1px solid #30363d' : '1px solid #e0e0e0', 
            backgroundImage: 'none',
            borderRadius: 8,
            boxShadow: mode === 'dark' ? 'none' : '0 2px 10px rgba(0,0,0,0.05)',
            display: 'flex',
            flexDirection: 'column'
          } 
        } 
      },
      MuiPaper: { styleOverrides: { root: { backgroundImage: 'none' } } },
      MuiTableCell: { styleOverrides: { root: { fontSize: '0.90rem', padding: '8px 16px' } } },
      MuiButton: { styleOverrides: { root: { textTransform: 'none', fontWeight: 600 } } }
    }
  }), [mode]);

  const chartColors = {
    grid: mode === 'dark' ? '#333' : '#e0e0e0',
    text: mode === 'dark' ? '#aaa' : '#555',
    tooltipBg: mode === 'dark' ? '#161b22' : '#ffffff',
    tooltipBorder: mode === 'dark' ? '#30363d' : '#ccc',
    baselineBar: mode === 'dark' ? '#888' : '#999',
    fitnessLine: mode === 'dark' ? '#ff5555' : '#d32f2f'
  };

  useEffect(() => {
    if (consoleEndRef.current) {
      consoleEndRef.current.scrollIntoView({ behavior: 'auto', block: 'nearest' });
    }
  }, [logs]);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("https://stu-attendance.nitrr.ac.in/fairness-api/api/analyze-columns", { method: "POST", body: formData });
      const data = await response.json();
      setFileId(data.file_id);
      setColumns(data.columns);
      setLogs(prev => [...prev, `[System]: Loaded dataset containing ${data.row_count} entries.`]);
    } catch (err) {
      setLogs(prev => [...prev, `[Error]: Failed uploading data.`]);
    }
  };

  const executePipeline = () => {
    if (!fileId || !targetCol || !sa1Col || !sa2Col) return alert("Map all features before starting.");
    
    setIsRunning(true);
    setProgress(0);
    setExportId(null);
    setOptimizedMetrics(null);
    setLiveGraphData([]);
    setFinalComparisonData([]);
    setExplainData([]);
    setLogs(prev => [...prev, "[System]: Establishing WebSocket pipelines..."]);

    const ws = new WebSocket("ws://localhost:8000/ws/train");

    ws.onopen = () => {
      ws.send(JSON.stringify({
        file_id: fileId, target_col: targetCol, sa1_col: sa1Col, sa2_col: sa2Col,
        pop_size: popSize, generations: generations, model_type: modelType
      }));
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.status === "log") {
        setLogs(prev => [...prev, `[Engine]: ${data.message}`]);
      } 
      else if (data.status === "baseline") {
        setBaselineMetrics(data.metrics);
        setLogs(prev => [...prev, "[System]: Base model complete. Starting optimization..."]);
      } 
      else if (data.status === "progress") {
        setProgress((data.generation / data.total_generations) * 100);
        setEta(data.eta);
        setOptimizedMetrics(data.metrics);
        
        setLiveGraphData(prev => [...prev, {
          generation: data.generation,
          di_sa1: (data.metrics.di1 !== undefined ? data.metrics.di1 : data.metrics.di_sa1).toFixed(3),
          di_sa2: (data.metrics.di2 !== undefined ? data.metrics.di2 : data.metrics.di_sa2).toFixed(3),
          fitness: data.metrics.fitness ? data.metrics.fitness.toFixed(3) : 0
        }]);

        setLogs(prev => [...prev, `[GA Loop]: ${data.message}`]);
      } 
      else if (data.status === "complete") {
        setProgress(100);
        setExportId(data.export_id);
        setIsRunning(false);
        setLogs(prev => [...prev, "[System]: Optimization finalized."]);
        ws.close();
      }
      else if (data.status === "error") {
        setLogs(prev => [...prev, `[Error]: ${data.message}`]);
        setIsRunning(false);
        ws.close();
      }
    };
  };

  const handleExplain = async () => {
    setIsExplaining(true);
    setExplainData([]);
    try {
      const res = await fetch(`http://localhost:8000/api/explain-${explainType}/${exportId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ instance_index: parseInt(explainIndex) || 0 })
      });
      const data = await res.json();
      
      let formattedData = [];
      if (explainType === 'shap') {
        formattedData = data.feature_names.map((name, i) => ({
          name: name,
          impact: data.impacts[i]
        })).sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact)).slice(0, 10);
      } else {
        formattedData = data.explanation.map(item => ({
          name: item[0],
          impact: item[1]
        }));
      }
      setExplainData(formattedData);
    } catch (err) {
      console.error("Explain error", err);
    }
    setIsExplaining(false);
  };

  useEffect(() => {
    if (!isRunning && baselineMetrics && optimizedMetrics) {
      setFinalComparisonData([
        { name: 'Accuracy (%)', Baseline: (baselineMetrics.acc || baselineMetrics.accuracy) * 100, Optimized: (optimizedMetrics.acc || optimizedMetrics.accuracy) * 100 },
        { name: `DI (${sa1Col})`, Baseline: baselineMetrics.di1 || baselineMetrics.di_sa1, Optimized: optimizedMetrics.di1 || optimizedMetrics.di_sa1 },
        { name: `DI (${sa2Col})`, Baseline: baselineMetrics.di2 || baselineMetrics.di_sa2, Optimized: optimizedMetrics.di2 || optimizedMetrics.di_sa2 },
      ]);
    }
  }, [isRunning, baselineMetrics, optimizedMetrics, sa1Col, sa2Col]);

  const getDIColor = (val) => (val >= 0.8 && val <= 1.25) ? theme.palette.success.main : 'inherit';
  const getDiffColor = (val) => (val >= -0.1 && val <= 0.1) ? theme.palette.success.main : 'inherit';

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ minHeight: '100vh', width: '100vw', p: { xs: 2, md: 4 }, boxSizing: 'border-box', display: 'flex', flexDirection: 'column', overflowX: 'hidden' }}>
        
        <Box sx={{ mb: 4, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Box display="flex" alignItems="center" gap={2}>
            <AssessmentIcon color="primary" sx={{ fontSize: 40 }} />
            <Box>
              <Typography variant="h4" component="h1">Intersectional Fairness Engine</Typography>
              <Typography variant="subtitle2" color="text.secondary" fontFamily="monospace">
                GA-Threshold Optimization Architecture // Status: {isRunning ? 'ACTIVE' : 'IDLE'}
              </Typography>
            </Box>
          </Box>
          <IconButton onClick={() => setMode(prev => prev === 'dark' ? 'light' : 'dark')} color="inherit" sx={{ border: 1, borderColor: 'divider', bgcolor: 'background.paper' }}>
            {mode === 'dark' ? <LightModeIcon /> : <DarkModeIcon />}
          </IconButton>
        </Box>

        <Box sx={{ display: 'flex', flexDirection: { xs: 'column', lg: 'row' }, flexGrow: 1, gap: 4 }}>
          
          <Box sx={{ width: { xs: '100%', lg: '380px', xl: '420px' }, flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 4 }}>
            <Card sx={{ flexGrow: 1 }}>
              <CardContent sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom color="primary">Configuration Matrix</Typography>
                
                <Button variant="outlined" component="label" startIcon={<UploadFileIcon />} fullWidth sx={{ mb: 3, py: 1.5 }}>
                  Mount CSV Dataset <input type="file" accept=".csv" hidden onChange={handleFileUpload} />
                </Button>

                {columns.length > 0 && (
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2.5 }}>
                    <FormControl fullWidth size="small">
                      <InputLabel sx={{ bgcolor: 'background.paper', px: 1 }}>Base Classifier</InputLabel>
                      <Select value={modelType} onChange={(e) => setModelType(e.target.value)}>
                        <MenuItem value="logistic_regression">Logistic Regression</MenuItem>
                        <MenuItem value="random_forest">Random Forest</MenuItem>
                        <MenuItem value="gradient_boosting">Gradient Boosting</MenuItem>
                        <MenuItem value="decision_tree">Decision Tree</MenuItem>
                      </Select>
                    </FormControl>
                    <FormControl fullWidth size="small">
                      <InputLabel sx={{ bgcolor: 'background.paper', px: 1 }}>Target Vector (Y)</InputLabel>
                      <Select value={targetCol} onChange={(e) => setTargetCol(e.target.value)}>
                        {columns.map(col => <MenuItem key={col} value={col}>{col}</MenuItem>)}
                      </Select>
                    </FormControl>
                    <FormControl fullWidth size="small">
                      <InputLabel sx={{ bgcolor: 'background.paper', px: 1 }}>Protected Attr 1</InputLabel>
                      <Select value={sa1Col} onChange={(e) => setSa1Col(e.target.value)}>
                        {columns.map(col => <MenuItem key={col} value={col}>{col}</MenuItem>)}
                      </Select>
                    </FormControl>
                    <FormControl fullWidth size="small">
                      <InputLabel sx={{ bgcolor: 'background.paper', px: 1 }}>Protected Attr 2</InputLabel>
                      <Select value={sa2Col} onChange={(e) => setSa2Col(e.target.value)}>
                        {columns.map(col => <MenuItem key={col} value={col}>{col}</MenuItem>)}
                      </Select>
                    </FormControl>
                  </Box>
                )}

                <Box sx={{ mt: 4, pt: 3, borderTop: 1, borderColor: 'divider' }}>
                  <Typography variant="body2" color="text.secondary">Population Size: <strong>{popSize}</strong></Typography>
                  <Slider value={popSize} min={50} max={300} step={10} onChange={(e, val) => setPopSize(val)} disabled={isRunning} size="small" />
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>Generations: <strong>{generations}</strong></Typography>
                  <Slider value={generations} min={10} max={100} step={5} onChange={(e, val) => setGenerations(val)} disabled={isRunning} size="small" />
                </Box>
                <Button variant="contained" color="primary" fullWidth startIcon={<PlayArrowIcon />} disabled={isRunning || !fileId} onClick={executePipeline} sx={{ mt: 3, py: 1.5, fontSize: '1.1rem' }}>
                  INITIALIZE TRAINING RUN
                </Button>
              </CardContent>
            </Card>
          </Box>

          <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', gap: 3, minWidth: 0 }}>
            
            <Paper variant="outlined" sx={{ bgcolor: mode === 'dark' ? '#0a0a0a' : '#f0f2f5', p: 2, display: 'flex', flexDirection: 'column', height: '220px', flexShrink: 0, borderRadius: 2 }}>
              <Box display="flex" alignItems="center" sx={{ mb: 1, borderBottom: 1, borderColor: 'divider', pb: 1, flexShrink: 0 }}>
                <TerminalIcon sx={{ mr: 1, color: mode === 'dark' ? '#00ff00' : '#0066cc', fontSize: '1.2rem' }} />
                <Typography variant="body2" fontFamily="monospace" color={mode === 'dark' ? '#00ff00' : '#0066cc'} fontWeight="bold">system.log</Typography>
              </Box>
              <Box sx={{ flexGrow: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                {logs.length === 0 ? <Typography variant="caption" color="text.disabled">Awaiting data...</Typography> : logs.map((log, idx) => (
                  <Typography key={idx} variant="caption" fontFamily="monospace" sx={{ color: log.includes('Error') ? '#ff5555' : (mode === 'dark' ? '#a8b2bd' : '#444') }}>{log}</Typography>
                ))}
                <div ref={consoleEndRef} />
              </Box>
            </Paper>

            <Box sx={{ mb: 1 }}>
              <Box display="flex" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
                <Typography variant="body2" fontWeight="bold" color="primary">Optimization Progress</Typography>
                <Typography variant="body2" fontFamily="monospace" color="text.secondary">ETA: {eta} | GEN: {liveGraphData.length}/{generations}</Typography>
              </Box>
              <LinearProgress variant="determinate" value={progress} sx={{ height: 6, borderRadius: 1 }} color={progress === 100 ? "success" : "primary"} />
            </Box>

            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3, flexGrow: 1 }}>
              <Card sx={{ display: 'flex', flexDirection: 'column', minHeight: '320px', flexGrow: 1 }}>
                <CardContent sx={{ p: 2.5, height: '100%', display: 'flex', flexDirection: 'column' }}>
                  <Typography variant="h6" color="primary" sx={{ display: 'flex', alignItems: 'center', fontSize: '1.1rem', mb: 2 }}><TimelineIcon sx={{ mr: 1 }}/> Fairness Metrics (DI)</Typography>
                  <Box sx={{ flexGrow: 1, width: '100%', minHeight: 0 }}>
                    {liveGraphData.length > 0 ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={liveGraphData} margin={{ top: 5, right: 20, left: -20, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
                          <XAxis dataKey="generation" stroke={chartColors.text} tick={{fontSize: 11}} />
                          <YAxis yAxisId="left" stroke={chartColors.text} tick={{fontSize: 11}} domain={[0, 'auto']} />
                          <RechartsTooltip contentStyle={{ backgroundColor: chartColors.tooltipBg, border: `1px solid ${chartColors.tooltipBorder}`, borderRadius: 4 }} />
                          <Legend wrapperStyle={{ fontSize: '0.85rem' }} />
                          <ReferenceLine yAxisId="left" y={1.0} stroke={theme.palette.error.main} strokeDasharray="3 3" />
                          <Line isAnimationActive={false} yAxisId="left" type="monotone" dataKey="di_sa1" name={`DI (${sa1Col})`} stroke={theme.palette.primary.main} dot={false} strokeWidth={2.5} />
                          <Line isAnimationActive={false} yAxisId="left" type="monotone" dataKey="di_sa2" name={`DI (${sa2Col})`} stroke={theme.palette.warning.main} dot={false} strokeWidth={2.5} />
                        </LineChart>
                      </ResponsiveContainer>
                    ) : <Box display="flex" height="100%" alignItems="center" justifyContent="center" border={`1px dashed ${chartColors.grid}`} borderRadius={2}><Typography variant="body2" color="text.secondary">Awaiting stream...</Typography></Box>}
                  </Box>
                </CardContent>
              </Card>

              <Card sx={{ display: 'flex', flexDirection: 'column', minHeight: '320px', flexGrow: 1 }}>
                <CardContent sx={{ p: 2.5, height: '100%', display: 'flex', flexDirection: 'column' }}>
                  <Typography variant="h6" color="error" sx={{ display: 'flex', alignItems: 'center', fontSize: '1.1rem', mb: 2 }}><TrendingDownIcon sx={{ mr: 1 }}/> GA Objective Cost (Fitness Penalty)</Typography>
                  <Box sx={{ flexGrow: 1, width: '100%', minHeight: 0 }}>
                    {liveGraphData.length > 0 ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={liveGraphData} margin={{ top: 5, right: 20, left: -10, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
                          <XAxis dataKey="generation" stroke={chartColors.text} tick={{fontSize: 11}} />
                          <YAxis yAxisId="left" stroke={chartColors.text} tick={{fontSize: 11}} domain={['auto', 'auto']} />
                          <RechartsTooltip contentStyle={{ backgroundColor: chartColors.tooltipBg, border: `1px solid ${chartColors.tooltipBorder}`, borderRadius: 4 }} />
                          <Legend wrapperStyle={{ fontSize: '0.85rem' }} />
                          <Line isAnimationActive={false} yAxisId="left" type="monotone" dataKey="fitness" name="Objective Cost" stroke={chartColors.fitnessLine} dot={false} strokeWidth={2.5} />
                        </LineChart>
                      </ResponsiveContainer>
                    ) : <Box display="flex" height="100%" alignItems="center" justifyContent="center" border={`1px dashed ${chartColors.grid}`} borderRadius={2}><Typography variant="body2" color="text.secondary">Awaiting stream...</Typography></Box>}
                  </Box>
                </CardContent>
              </Card>
            </Box>

            {exportId && baselineMetrics && optimizedMetrics && (
              <Card sx={{ border: `1px solid ${theme.palette.success.main}`, boxShadow: `0 0 15px ${theme.palette.success.main}22`, mt: 2 }}>
                <CardContent sx={{ p: 3 }}>
                  <Typography variant="h5" color="success.main" gutterBottom sx={{ mb: 3 }}>Optimization Finalized</Typography>
                  
                  <Grid container spacing={4} sx={{ mb: 3 }}>
                    <Grid item xs={12} xl={6}>
                      <TableContainer component={Paper} elevation={0} sx={{ border: 1, borderColor: 'divider' }}>
                        <Table size="small">
                          <TableHead sx={{ bgcolor: mode === 'dark' ? '#21262d' : '#f4f6f8' }}>
                            <TableRow>
                              <TableCell sx={{ fontWeight: 'bold' }}>Performance Metrics</TableCell>
                              <TableCell align="right"><strong>Baseline</strong></TableCell>
                              <TableCell align="right"><strong>Optimized</strong></TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            <TableRow>
                              <TableCell>Accuracy</TableCell>
                              <TableCell align="right">{((baselineMetrics.acc || baselineMetrics.accuracy) * 100).toFixed(2)}%</TableCell>
                              <TableCell align="right">{((optimizedMetrics.acc || optimizedMetrics.accuracy) * 100).toFixed(2)}%</TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell>AUROC</TableCell>
                              <TableCell align="right">{(baselineMetrics.auroc || 0).toFixed(4)}</TableCell>
                              <TableCell align="right">{(optimizedMetrics.auroc || 0).toFixed(4)}</TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell>AUPRC</TableCell>
                              <TableCell align="right">{(baselineMetrics.auprc || 0).toFixed(4)}</TableCell>
                              <TableCell align="right">{(optimizedMetrics.auprc || 0).toFixed(4)}</TableCell>
                            </TableRow>
                            
                            <TableRow sx={{ bgcolor: mode === 'dark' ? '#1c2128' : '#fafafa' }}>
                              <TableCell colSpan={3} sx={{ fontWeight: 'bold', color: 'primary.main', pt: 2 }}>Fairness: {sa1Col}</TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell>Disparate Impact (DI)</TableCell>
                              <TableCell align="right">{(baselineMetrics.di1 || baselineMetrics.di_sa1).toFixed(4)}</TableCell>
                              <TableCell align="right" sx={{ fontWeight: 'bold', color: getDIColor(optimizedMetrics.di1 || optimizedMetrics.di_sa1) }}>{(optimizedMetrics.di1 || optimizedMetrics.di_sa1).toFixed(4)}</TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell>Stat Parity Diff (SPD)</TableCell>
                              <TableCell align="right">{(baselineMetrics.spd1 || 0).toFixed(4)}</TableCell>
                              <TableCell align="right" sx={{ fontWeight: 'bold', color: getDiffColor(optimizedMetrics.spd1 || 0) }}>{(optimizedMetrics.spd1 || 0) > 0 ? '+':''}{(optimizedMetrics.spd1 || 0).toFixed(4)}</TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell>Avg Odds Diff (AOD)</TableCell>
                              <TableCell align="right">{(baselineMetrics.aod1 || 0).toFixed(4)}</TableCell>
                              <TableCell align="right" sx={{ fontWeight: 'bold', color: getDiffColor(optimizedMetrics.aod1 || 0) }}>{(optimizedMetrics.aod1 || 0) > 0 ? '+':''}{(optimizedMetrics.aod1 || 0).toFixed(4)}</TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell>Equal Opp Diff (EOD)</TableCell>
                              <TableCell align="right">{(baselineMetrics.eod1 || 0).toFixed(4)}</TableCell>
                              <TableCell align="right" sx={{ fontWeight: 'bold', color: getDiffColor(optimizedMetrics.eod1 || 0) }}>{(optimizedMetrics.eod1 || 0) > 0 ? '+':''}{(optimizedMetrics.eod1 || 0).toFixed(4)}</TableCell>
                            </TableRow>

                            <TableRow sx={{ bgcolor: mode === 'dark' ? '#1c2128' : '#fafafa' }}>
                              <TableCell colSpan={3} sx={{ fontWeight: 'bold', color: theme.palette.warning.main, pt: 2 }}>Fairness: {sa2Col}</TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell>Disparate Impact (DI)</TableCell>
                              <TableCell align="right">{(baselineMetrics.di2 || baselineMetrics.di_sa2).toFixed(4)}</TableCell>
                              <TableCell align="right" sx={{ fontWeight: 'bold', color: getDIColor(optimizedMetrics.di2 || optimizedMetrics.di_sa2) }}>{(optimizedMetrics.di2 || optimizedMetrics.di_sa2).toFixed(4)}</TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell>Stat Parity Diff (SPD)</TableCell>
                              <TableCell align="right">{(baselineMetrics.spd2 || 0).toFixed(4)}</TableCell>
                              <TableCell align="right" sx={{ fontWeight: 'bold', color: getDiffColor(optimizedMetrics.spd2 || 0) }}>{(optimizedMetrics.spd2 || 0) > 0 ? '+':''}{(optimizedMetrics.spd2 || 0).toFixed(4)}</TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell>Avg Odds Diff (AOD)</TableCell>
                              <TableCell align="right">{(baselineMetrics.aod2 || 0).toFixed(4)}</TableCell>
                              <TableCell align="right" sx={{ fontWeight: 'bold', color: getDiffColor(optimizedMetrics.aod2 || 0) }}>{(optimizedMetrics.aod2 || 0) > 0 ? '+':''}{(optimizedMetrics.aod2 || 0).toFixed(4)}</TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell>Equal Opp Diff (EOD)</TableCell>
                              <TableCell align="right">{(baselineMetrics.eod2 || 0).toFixed(4)}</TableCell>
                              <TableCell align="right" sx={{ fontWeight: 'bold', color: getDiffColor(optimizedMetrics.eod2 || 0) }}>{(optimizedMetrics.eod2 || 0) > 0 ? '+':''}{(optimizedMetrics.eod2 || 0).toFixed(4)}</TableCell>
                            </TableRow>
                          </TableBody>
                        </Table>
                      </TableContainer>
                    </Grid>

                    <Grid item xs={12} xl={6}>
                      <Box sx={{ height: 180, width: '100%' }}>
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={finalComparisonData} margin={{ top: 5, right: 10, left: -25, bottom: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
                            <XAxis dataKey="name" stroke={chartColors.text} tick={{ fontSize: 11 }} />
                            <YAxis stroke={chartColors.text} tick={{ fontSize: 11 }} />
                            <RechartsTooltip contentStyle={{ backgroundColor: chartColors.tooltipBg, border: `1px solid ${chartColors.tooltipBorder}` }} />
                            <Legend wrapperStyle={{ fontSize: '0.9rem' }} />
                            <Bar dataKey="Baseline" fill={chartColors.baselineBar} radius={[2, 2, 0, 0]} />
                            <Bar dataKey="Optimized" fill={theme.palette.success.main} radius={[2, 2, 0, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      </Box>
                    </Grid>
                  </Grid>

                  <Divider sx={{ my: 3 }} />

                  <Grid container spacing={2}>
                    <Grid item xs={12} sm={6}>
                      <Button
                        variant="contained" color="success" fullWidth startIcon={<DownloadIcon />}
                        onClick={() => window.open(`http://localhost:8000/api/download-model/${exportId}`, '_blank')}
                        sx={{ py: 1.5 }}
                      >
                        Export Weights (.JSON)
                      </Button>
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <Button
                        variant="outlined" color="primary" fullWidth startIcon={<ImageIcon />}
                        onClick={() => window.open(`http://localhost:8000/api/report-image/${exportId}`, '_blank')}
                        sx={{ py: 1.5, bgcolor: 'background.paper' }}
                      >
                        Download Report (.PNG)
                      </Button>
                    </Grid>
                  </Grid>

                </CardContent>
              </Card>
            )}

            {exportId && (
              <Card sx={{ border: `1px solid ${theme.palette.primary.main}`, mt: 2 }}>
                <CardContent sx={{ p: 3 }}>
                  <Typography variant="h5" color="primary.main" gutterBottom>Local Instance Auditor (LIME / SHAP)</Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                    Audit the base classifier to see exactly which features drove the prediction up or down for a specific test subject.
                  </Typography>
                  
                  <Grid container spacing={2} alignItems="center" sx={{ mb: 3 }}>
                    <Grid item>
                      <FormControl size="small" sx={{ width: 140 }}>
                        <InputLabel sx={{ bgcolor: 'background.paper', px: 1 }}>Explainer Tool</InputLabel>
                        <Select value={explainType} onChange={(e) => setExplainType(e.target.value)}>
                          <MenuItem value="lime">LIME</MenuItem>
                          <MenuItem value="shap">SHAP</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid item>
                      <TextField 
                        size="small" 
                        label="Test Set Row ID" 
                        type="number" 
                        value={explainIndex} 
                        onChange={(e) => setExplainIndex(e.target.value)}
                        InputLabelProps={{ sx: { bgcolor: 'background.paper', px: 1 } }}
                        sx={{ width: 140 }}
                        inputProps={{ min: 0 }}
                      />
                    </Grid>
                    <Grid item>
                      <Button variant="contained" color="primary" onClick={handleExplain} disabled={isExplaining} sx={{ py: 1 }}>
                        {isExplaining ? 'Generating...' : 'Generate Explanation'}
                      </Button>
                    </Grid>
                  </Grid>

                  {explainData.length > 0 && (
                    <Box sx={{ height: 350, width: '100%', mt: 2 }}>
                      <Typography variant="body2" fontWeight="bold" textAlign="center" mb={2}>
                        Feature Impacts (Green = Pushes toward Outcome 1, Red = Pushes toward Outcome 0)
                      </Typography>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart layout="vertical" data={explainData} margin={{ top: 5, right: 30, left: 120, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
                          <XAxis type="number" stroke={chartColors.text} tick={{ fontSize: 11 }} />
                          <YAxis dataKey="name" type="category" stroke={chartColors.text} tick={{ fontSize: 11 }} width={140} />
                          <RechartsTooltip contentStyle={{ backgroundColor: chartColors.tooltipBg, border: `1px solid ${chartColors.tooltipBorder}`, borderRadius: 4 }} />
                          <Bar dataKey="impact" radius={[0, 4, 4, 0]}>
                            {explainData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={entry.impact > 0 ? theme.palette.success.main : theme.palette.error.main} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </Box>
                  )}
                </CardContent>
              </Card>
            )}
          </Box>
        </Box>
      </Box>
    </ThemeProvider>
  );
}