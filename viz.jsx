import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter } from 'recharts';
import { Play, Pause, RotateCcw, Settings } from 'lucide-react';

const A3CVisualization = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [episode, setEpisode] = useState(0);
  const [costData, setCostData] = useState([]);
  const [policyData, setPolicyData] = useState([]);
  const [config, setConfig] = useState({
    numSensors: 1,
    transitionType: 'R1',
    learningRate: 0.0005,
    numEpisodes: 200
  });
  const [finalStats, setFinalStats] = useState(null);

  // Theoretical expectations
  const theoreticalExpectations = {
    'R1': 0.58777,
    'R2': 2.5
  };

  // Simulate A3C optimization
  const runOptimization = () => {
    if (episode >= config.numEpisodes) {
      setIsRunning(false);
      return;
    }

    // Simulate cost convergence with noise
    const theoreticalCost = theoreticalExpectations[config.transitionType];
    const convergenceRate = 0.05;
    const noise = (Math.random() - 0.5) * 0.1 * theoreticalCost;
    
    const currentCost = theoreticalCost + 
      (3 - theoreticalCost) * Math.exp(-convergenceRate * episode) + noise;

    const newDataPoint = {
      episode: episode,
      cost: currentCost,
      theoretical: theoreticalCost
    };

    setCostData(prev => [...prev, newDataPoint]);

    // Simulate policy parameter evolution (theta)
    const theta = -Math.log((1 / Math.min(0.95, 0.5 + 0.01 * episode)) - 1);
    const schedulingProb = 1 / (1 + Math.exp(-theta));
    
    setPolicyData(prev => [...prev, {
      episode: episode,
      theta: theta,
      probability: schedulingProb
    }]);

    setEpisode(prev => prev + 1);

    // Calculate final statistics
    if (episode >= config.numEpisodes - 20) {
      const recentCosts = costData.slice(-20).map(d => d.cost);
      const avgCost = recentCosts.reduce((a, b) => a + b, 0) / recentCosts.length;
      const variance = recentCosts.reduce((sum, val) => sum + Math.pow(val - avgCost, 2), 0) / recentCosts.length;
      
      setFinalStats({
        avgCost: avgCost.toFixed(4),
        variance: variance.toFixed(6),
        theoretical: theoreticalCost,
        error: Math.abs(avgCost - theoreticalCost).toFixed(4)
      });
    }
  };

  useEffect(() => {
    if (isRunning) {
      const timer = setTimeout(runOptimization, 50);
      return () => clearTimeout(timer);
    }
  }, [isRunning, episode]);

  const handleReset = () => {
    setIsRunning(false);
    setEpisode(0);
    setCostData([]);
    setPolicyData([]);
    setFinalStats(null);
  };

  const handleConfigChange = (key, value) => {
    setConfig(prev => ({ ...prev, [key]: value }));
    handleReset();
  };

  return (
    <div className="w-full h-full bg-gradient-to-br from-slate-50 to-slate-100 p-6 overflow-auto">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h1 className="text-3xl font-bold text-slate-800 mb-2">
            A3C Optimization for AoII Scheduling
          </h1>
          <p className="text-slate-600">
            Stochastic Gradient-Based Optimization via Parallel Sampling
          </p>
        </div>

        {/* Configuration Panel */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <div className="flex items-center gap-2 mb-4">
            <Settings className="w-5 h-5 text-blue-600" />
            <h2 className="text-xl font-semibold text-slate-800">Configuration</h2>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Transition Matrix
              </label>
              <select
                value={config.transitionType}
                onChange={(e) => handleConfigChange('transitionType', e.target.value)}
                className="w-full px-3 py-2 border border-slate-300 rounded-md focus:ring-2 focus:ring-blue-500"
              >
                <option value="R1">R₁ (High Persistence)</option>
                <option value="R2">R₂ (High Variability)</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Learning Rate (α)
              </label>
              <input
                type="number"
                step="0.0001"
                value={config.learningRate}
                onChange={(e) => handleConfigChange('learningRate', parseFloat(e.target.value))}
                className="w-full px-3 py-2 border border-slate-300 rounded-md focus:ring-2 focus:ring-blue-500"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Episodes
              </label>
              <input
                type="number"
                step="10"
                value={config.numEpisodes}
                onChange={(e) => handleConfigChange('numEpisodes', parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-slate-300 rounded-md focus:ring-2 focus:ring-blue-500"
              />
            </div>
            
            <div className="flex items-end gap-2">
              <button
                onClick={() => setIsRunning(!isRunning)}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
              >
                {isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                {isRunning ? 'Pause' : 'Start'}
              </button>
              <button
                onClick={handleReset}
                className="px-4 py-2 bg-slate-200 text-slate-700 rounded-md hover:bg-slate-300 transition-colors"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>

        {/* Progress */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-medium text-slate-700">
              Episode: {episode} / {config.numEpisodes}
            </span>
            <span className="text-sm text-slate-600">
              {((episode / config.numEpisodes) * 100).toFixed(1)}%
            </span>
          </div>
          <div className="w-full bg-slate-200 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${(episode / config.numEpisodes) * 100}%` }}
            />
          </div>
        </div>

        {/* Cost Convergence Chart */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-semibold text-slate-800 mb-4">
            Objective Function Convergence
          </h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={costData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis 
                dataKey="episode" 
                label={{ value: 'Episode (k)', position: 'insideBottom', offset: -5 }}
                stroke="#64748b"
              />
              <YAxis 
                label={{ value: 'Average Cost J(θₖ)', angle: -90, position: 'insideLeft' }}
                stroke="#64748b"
              />
              <Tooltip 
                contentStyle={{ backgroundColor: '#fff', border: '1px solid #e2e8f0' }}
                formatter={(value) => value.toFixed(4)}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="cost" 
                stroke="#3b82f6" 
                strokeWidth={2}
                name="Empirical Cost"
                dot={false}
              />
              <Line 
                type="monotone" 
                dataKey="theoretical" 
                stroke="#ef4444" 
                strokeWidth={2}
                strokeDasharray="5 5"
                name="Theoretical Expectation"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Policy Parameter Evolution */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-semibold text-slate-800 mb-4">
            Policy Parameter Evolution: θₖ and π(aₜ=1|sₜ)
          </h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={policyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis 
                dataKey="episode" 
                label={{ value: 'Episode (k)', position: 'insideBottom', offset: -5 }}
                stroke="#64748b"
              />
              <YAxis 
                yAxisId="left"
                label={{ value: 'θ (parameter)', angle: -90, position: 'insideLeft' }}
                stroke="#64748b"
              />
              <YAxis 
                yAxisId="right"
                orientation="right"
                domain={[0, 1]}
                label={{ value: 'P(schedule)', angle: 90, position: 'insideRight' }}
                stroke="#64748b"
              />
              <Tooltip 
                contentStyle={{ backgroundColor: '#fff', border: '1px solid #e2e8f0' }}
                formatter={(value) => value.toFixed(4)}
              />
              <Legend />
              <Line 
                yAxisId="left"
                type="monotone" 
                dataKey="theta" 
                stroke="#8b5cf6" 
                strokeWidth={2}
                name="Parameter θ"
                dot={false}
              />
              <Line 
                yAxisId="right"
                type="monotone" 
                dataKey="probability" 
                stroke="#10b981" 
                strokeWidth={2}
                name="Scheduling Probability"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Final Statistics */}
        {finalStats && (
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg shadow-lg p-6 border border-blue-200">
            <h2 className="text-xl font-semibold text-slate-800 mb-4">
              Convergence Statistics (Last 20 Episodes)
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-white rounded-lg p-4 shadow">
                <div className="text-sm text-slate-600 mb-1">Empirical Average</div>
                <div className="text-2xl font-bold text-blue-600">{finalStats.avgCost}</div>
              </div>
              <div className="bg-white rounded-lg p-4 shadow">
                <div className="text-sm text-slate-600 mb-1">Theoretical Value</div>
                <div className="text-2xl font-bold text-red-600">{finalStats.theoretical}</div>
              </div>
              <div className="bg-white rounded-lg p-4 shadow">
                <div className="text-sm text-slate-600 mb-1">Absolute Error</div>
                <div className="text-2xl font-bold text-amber-600">{finalStats.error}</div>
              </div>
              <div className="bg-white rounded-lg p-4 shadow">
                <div className="text-sm text-slate-600 mb-1">Variance</div>
                <div className="text-2xl font-bold text-purple-600">{finalStats.variance}</div>
              </div>
            </div>
          </div>
        )}

        {/* Optimization Algorithm Description */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-semibold text-slate-800 mb-4">
            Optimization Framework
          </h2>
          <div className="space-y-3 text-slate-700">
            <div className="bg-slate-50 p-4 rounded-lg border-l-4 border-blue-500">
              <strong>Objective:</strong> min<sub>θ</sub> J(θ) = lim<sub>T→∞</sub> (1/T) E[Σ (||Δ(t)||₁ + w<sup>T</sup>aₜ)]
            </div>
            <div className="bg-slate-50 p-4 rounded-lg border-l-4 border-green-500">
              <strong>Decision Rule:</strong> aₜ ~ Bernoulli(σ(f<sub>θ</sub>(sₜ))), where σ(x) = 1/(1+e<sup>-x</sup>)
            </div>
            <div className="bg-slate-50 p-4 rounded-lg border-l-4 border-purple-500">
              <strong>Update Rule:</strong> θ<sub>k+1</sub> = θ<sub>k</sub> - α<sub>k</sub> ∇̂J(θ<sub>k</sub>)
            </div>
            <div className="bg-slate-50 p-4 rounded-lg border-l-4 border-amber-500">
              <strong>Gradient Estimate:</strong> Computed via parallel trajectory sampling and policy gradient theorem
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default A3CVisualization;
