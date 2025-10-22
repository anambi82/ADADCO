import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import './DataVisualization.css';

const DataVisualization = () => {
    const [analysisData, setAnalysisData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const fetchAnalysisData = async () => {
        setLoading(true);
        setError('');
        try {
            const response = await axios.get('http://localhost:8000/analysis/summary');
            if (response.data.available) {
                setAnalysisData(response.data);
            } else {
                setError('No analysis data available. Please upload and analyze a file first.');
            }
        } catch (err) {
            setError(`Error fetching analysis data: ${err.response?.data?.error || err.message}`);
        } finally {
            setLoading(false);
        }
    };

    const fetchAttackData = async () => {
        try {
            const response = await axios.get('http://localhost:8000/analysis/attacks');
            if (response.data.available) {
                setAnalysisData(prev => ({ ...prev, attacks: response.data.attacks }));
            }
        } catch (err) {
            console.error('Error fetching attack data:', err);
        }
    };

    useEffect(() => {
        // Always fetch fresh data when component mounts
        fetchAnalysisData();
    }, []); // Empty dependency array means this runs once on mount

    useEffect(() => {
        if (analysisData && !analysisData.attacks) {
            fetchAttackData();
        }
    }, [analysisData]);

    // Convert attack data for charts
    const getAttackChartData = () => {
        if (!analysisData?.attacks) return [];
        return analysisData.attacks.map(attack => ({
            name: attack.attack_type,
            count: attack.count,
            percentage: attack.percentage
        }));
    };

    const getTimelineData = () => {
        if (!analysisData?.summary) return [];
        return [
            { name: 'Benign', count: analysisData.summary.benign_count || 0 },
            { name: 'Anomalies', count: analysisData.summary.anomaly_count || 0 }
        ];
    };

    const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D', '#FFC658'];

    return (
        <div className="data-visualization-container">
            <h2>Network Traffic Analysis Dashboard</h2>

            <button onClick={fetchAnalysisData} disabled={loading} className="refresh-button">
                {loading ? 'Loading...' : 'Refresh Analysis'}
            </button>

            {error && <div className="error-message">{error}</div>}

            {analysisData && (
                <div className="analysis-summary">
                    <h3>Analysis Summary</h3>
                    <div className="summary-cards">
                        <div className="summary-card">
                            <h4>Total Samples</h4>
                            <p>{analysisData.summary?.total_samples || 0}</p>
                        </div>
                        <div className="summary-card">
                            <h4>Anomalies Detected</h4>
                            <p>{analysisData.summary?.anomaly_count || 0}</p>
                        </div>
                        <div className="summary-card">
                            <h4>Anomaly Rate</h4>
                            <p>{analysisData.summary?.anomaly_rate || 0}%</p>
                        </div>
                        <div className="summary-card">
                            <h4>Severity</h4>
                            <p className={`severity-${analysisData.summary?.severity?.toLowerCase() || 'low'}`}>
                                {analysisData.summary?.severity || 'LOW'}
                            </p>
                        </div>
                    </div>
                </div>
            )}

            <div className="charts-container">
                {analysisData?.attacks && analysisData.attacks.length > 0 && (
                    <>
                        <div className="chart-wrapper">
                            <h3>Attack Type Distribution</h3>
                            <ResponsiveContainer width="100%" height={300}>
                                <BarChart data={getAttackChartData()}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                                    <YAxis />
                                    <Tooltip />
                                    <Legend />
                                    <Bar dataKey="count" fill="#8884d8" />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>

                        <div className="chart-wrapper">
                            <h3>Attack Types Pie Chart</h3>
                            <ResponsiveContainer width="100%" height={300}>
                                <PieChart>
                                    <Pie
                                        data={getAttackChartData()}
                                        cx="50%"
                                        cy="50%"
                                        labelLine={false}
                                        label={({ name, percentage }) => `${name}: ${percentage.toFixed(1)}%`}
                                        outerRadius={80}
                                        fill="#8884d8"
                                        dataKey="count"
                                    >
                                        {getAttackChartData().map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                        ))}
                                    </Pie>
                                    <Tooltip />
                                </PieChart>
                            </ResponsiveContainer>
                        </div>
                    </>
                )}

                {analysisData?.summary && (
                    <div className="chart-wrapper">
                        <h3>Benign vs Anomalies</h3>
                        <ResponsiveContainer width="100%" height={300}>
                            <BarChart data={getTimelineData()}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="name" />
                                <YAxis />
                                <Tooltip />
                                <Legend />
                                <Bar dataKey="count" fill="#82ca9d" />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                )}
            </div>

            {analysisData && (
                <div className="data-preview">
                    <h3>Analysis Details</h3>
                    <pre>{JSON.stringify(analysisData, null, 2)}</pre>
                </div>
            )}
        </div>
    );
};

export default DataVisualization;
