import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import './DataVisualization.css';

const DataVisualization = ({ selectedRecord }) => {
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

    const fetchConfidenceData = async () => {
        try {
            const response = await axios.get('http://localhost:8000/analysis/confidence');
            if (response.data.available) {
                setAnalysisData(prev => ({ ...prev, confidence: response.data }));
            }
        } catch (err) {
            console.error('Error fetching confidence data:', err);
        }
    };

    useEffect(() => {
        // If a record is selected, use it; otherwise fetch fresh data
        if (selectedRecord) {
            // Calculate confidence statistics if missing
            let confidenceData = selectedRecord.confidence;
            if (confidenceData && !confidenceData.statistics && confidenceData.confidence_by_attack) {
                const confidences = confidenceData.confidence_by_attack.map(c => c.average_confidence);
                if (confidences.length > 0) {
                    confidenceData = {
                        ...confidenceData,
                        statistics: {
                            mean: confidences.reduce((a, b) => a + b, 0) / confidences.length,
                            median: confidences.sort()[Math.floor(confidences.length / 2)],
                            min: Math.min(...confidences),
                            max: Math.max(...confidences)
                        }
                    };
                }
            }
            
            setAnalysisData({
                summary: selectedRecord.summary,
                attacks: selectedRecord.attacks,
                confidence: confidenceData,
                timestamp: selectedRecord.timestamp,
                available: true
            });
            setError('');
        } else {
            fetchAnalysisData();
        }
    }, [selectedRecord]); // Runs when selectedRecord changes

    useEffect(() => {
        // Only fetch additional data if we're not using a selected record
        if (!selectedRecord) {
            if (analysisData && !analysisData.attacks) {
                fetchAttackData();
            }
            if (analysisData && !analysisData.confidence) {
                fetchConfidenceData();
            }
        }
    }, [analysisData, selectedRecord]);

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
            
            {selectedRecord && (
                <div className="record-indicator">
                    📄 Viewing saved record from {new Date(selectedRecord.timestamp).toLocaleString()}
                </div>
            )}

            {!selectedRecord && (
                <button onClick={fetchAnalysisData} disabled={loading} className="refresh-button">
                    {loading ? 'Loading...' : 'Refresh Analysis'}
                </button>
            )}

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
                    <div className="analysis-details-grid">
                        {analysisData.summary && (
                            <div className="details-section">
                                <h4>Summary Information</h4>
                                <div className="detail-row">
                                    <span className="detail-label">Total Samples:</span>
                                    <span className="detail-value">{analysisData.summary.total_samples || 0}</span>
                                </div>
                                <div className="detail-row">
                                    <span className="detail-label">Benign Count:</span>
                                    <span className="detail-value">{analysisData.summary.benign_count || 0}</span>
                                </div>
                                <div className="detail-row">
                                    <span className="detail-label">Anomaly Count:</span>
                                    <span className="detail-value">{analysisData.summary.anomaly_count || 0}</span>
                                </div>
                                <div className="detail-row">
                                    <span className="detail-label">Anomaly Rate:</span>
                                    <span className="detail-value">{analysisData.summary.anomaly_rate || 0}%</span>
                                </div>
                                <div className="detail-row">
                                    <span className="detail-label">Severity Level:</span>
                                    <span className={`detail-value severity-badge severity-${analysisData.summary.severity?.toLowerCase() || 'low'}`}>
                                        {analysisData.summary.severity || 'LOW'}
                                    </span>
                                </div>
                            </div>
                        )}

                        {analysisData.attacks && analysisData.attacks.length > 0 && (
                            <div className="details-section">
                                <h4>Attack Breakdown</h4>
                                <div className="attack-list">
                                    {analysisData.attacks.map((attack, index) => {
                                        const confidenceData = analysisData.confidence?.confidence_by_attack?.find(
                                            c => c.attack_type === attack.attack_type
                                        );
                                        const confidence = confidenceData?.average_confidence;

                                        return (
                                            <div key={index} className="attack-item">
                                                <div className="attack-header">
                                                    <span className="attack-type">{attack.attack_type}</span>
                                                    <span className="attack-count">{attack.count} occurrences</span>
                                                </div>
                                                <div className="attack-percentage">
                                                    <div className="percentage-bar">
                                                        <div
                                                            className="percentage-fill"
                                                            style={{ width: `${attack.percentage}%`, backgroundColor: COLORS[index % COLORS.length] }}
                                                        ></div>
                                                    </div>
                                                    <span className="percentage-text">{attack.percentage.toFixed(2)}%</span>
                                                </div>
                                                {confidence !== undefined && (
                                                    <div className="attack-confidence">
                                                        <span className="confidence-label">Confidence:</span>
                                                        <span className="confidence-value">{(confidence * 100).toFixed(2)}%</span>
                                                    </div>
                                                )}
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        )}

                        {analysisData.confidence?.statistics && (
                            <div className="details-section">
                                <h4>Confidence Statistics</h4>
                                <div className="detail-row">
                                    <span className="detail-label">Mean Confidence:</span>
                                    <span className="detail-value">{(analysisData.confidence.statistics.mean * 100).toFixed(2)}%</span>
                                </div>
                                <div className="detail-row">
                                    <span className="detail-label">Median Confidence:</span>
                                    <span className="detail-value">{(analysisData.confidence.statistics.median * 100).toFixed(2)}%</span>
                                </div>
                                <div className="detail-row">
                                    <span className="detail-label">Min Confidence:</span>
                                    <span className="detail-value">{(analysisData.confidence.statistics.min * 100).toFixed(2)}%</span>
                                </div>
                                <div className="detail-row">
                                    <span className="detail-label">Max Confidence:</span>
                                    <span className="detail-value">{(analysisData.confidence.statistics.max * 100).toFixed(2)}%</span>
                                </div>
                            </div>
                        )}

                        {analysisData.timestamp && (
                            <div className="details-section">
                                <h4>Analysis Metadata</h4>
                                <div className="detail-row">
                                    <span className="detail-label">Analysis Date:</span>
                                    <span className="detail-value">{new Date(analysisData.timestamp).toLocaleString()}</span>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

export default DataVisualization;
