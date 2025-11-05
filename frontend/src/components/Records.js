import React, { useState, useEffect } from 'react';
import { getRecords, deleteRecord, clearAllRecords } from '../utils/cookieManager';
import './Records.css';

const Records = ({ onSelectRecord }) => {
    const [records, setRecords] = useState([]);
    const [selectedRecordId, setSelectedRecordId] = useState(null);

    useEffect(() => {
        loadRecords();
    }, []);

    const loadRecords = () => {
        const storedRecords = getRecords();
        setRecords(storedRecords);
    };

    const handleSelectRecord = (record) => {
        setSelectedRecordId(record.id);
        onSelectRecord(record);
    };

    const handleDeleteRecord = (e, id) => {
        e.stopPropagation();
        if (window.confirm('Are you sure you want to delete this record?')) {
            deleteRecord(id);
            loadRecords();
            if (selectedRecordId === id) {
                setSelectedRecordId(null);
                onSelectRecord(null);
            }
        }
    };

    const handleClearAll = () => {
        if (window.confirm('Are you sure you want to delete all records? This cannot be undone.')) {
            clearAllRecords();
            loadRecords();
            setSelectedRecordId(null);
            onSelectRecord(null);
        }
    };

    const formatDate = (isoString) => {
        const date = new Date(isoString);
        return date.toLocaleString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    };

    const getSeverityClass = (severity) => {
        if (!severity) return 'severity-low';
        return `severity-${severity.toLowerCase()}`;
    };

    return (
        <div className="records-container">
            <div className="records-header">
                <h2>Analysis Records</h2>
                {records.length > 0 && (
                    <button onClick={handleClearAll} className="clear-all-button">
                        Clear All Records
                    </button>
                )}
            </div>

            {records.length === 0 ? (
                <div className="no-records">
                    <p>No analysis records found.</p>
                    <p className="hint">Upload and analyze files to create records.</p>
                </div>
            ) : (
                <div className="records-list">
                    {records.map((record) => (
                        <div
                            key={record.id}
                            className={`record-card ${selectedRecordId === record.id ? 'selected' : ''}`}
                            onClick={() => handleSelectRecord(record)}
                        >
                            <div className="record-header">
                                <div className="record-title">
                                    <span className="file-icon">📄</span>
                                    <span className="file-name">{record.fileName}</span>
                                </div>
                                <button
                                    className="delete-button"
                                    onClick={(e) => handleDeleteRecord(e, record.id)}
                                    title="Delete record"
                                >
                                    🗑️
                                </button>
                            </div>

                            <div className="record-details">
                                <div className="record-date">
                                    <span className="detail-label">Date:</span>
                                    <span className="detail-value">{formatDate(record.timestamp)}</span>
                                </div>

                                {record.summary && (
                                    <>
                                        <div className="record-stats">
                                            <div className="stat-item">
                                                <span className="stat-label">Samples:</span>
                                                <span className="stat-value">{record.summary.total_samples || 0}</span>
                                            </div>
                                            <div className="stat-item">
                                                <span className="stat-label">Anomalies:</span>
                                                <span className="stat-value anomaly">{record.summary.anomaly_count || 0}</span>
                                            </div>
                                            <div className="stat-item">
                                                <span className="stat-label">Rate:</span>
                                                <span className="stat-value">{record.summary.anomaly_rate || 0}%</span>
                                            </div>
                                        </div>

                                        <div className="record-severity">
                                            <span className="detail-label">Severity:</span>
                                            <span className={`severity-badge ${getSeverityClass(record.summary.severity)}`}>
                                                {record.summary.severity || 'LOW'}
                                            </span>
                                        </div>
                                    </>
                                )}

                                {record.attacks && record.attacks.length > 0 && (
                                    <div className="record-attacks">
                                        <span className="detail-label">Attack Types:</span>
                                        <div className="attack-tags">
                                            {record.attacks.slice(0, 3).map((attack, index) => (
                                                <span key={index} className="attack-tag">
                                                    {attack.attack_type}
                                                </span>
                                            ))}
                                            {record.attacks.length > 3 && (
                                                <span className="attack-tag more">
                                                    +{record.attacks.length - 3} more
                                                </span>
                                            )}
                                        </div>
                                    </div>
                                )}
                            </div>

                            {selectedRecordId === record.id && (
                                <div className="selected-indicator">
                                    ✓ Selected
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            )}

            {records.length > 0 && (
                <div className="records-footer">
                    <p className="records-info">
                        Showing {records.length} of max 10 records. Records are stored locally in your browser cookies.
                    </p>
                </div>
            )}
        </div>
    );
};

export default Records;

