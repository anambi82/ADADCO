import React, { useState } from 'react';
import axios from 'axios';
import { saveRecord } from '../utils/cookieManager';
import './FileUpload.css';

const FileUpload = ({ onUploadSuccess }) => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [uploading, setUploading] = useState(false);
    const [message, setMessage] = useState('');

    const handleFileSelect = (event) => {
        setSelectedFile(event.target.files[0]);
        setMessage('');
    };

    const handleUpload = async () => {
        if (!selectedFile) {
            setMessage('Please select a file first');
            return;
        }

        const formData = new FormData();
        formData.append('file', selectedFile);

        setUploading(true);
        setMessage('');

        try {
            const response = await axios.post('http://localhost:8000/analyze_file', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });

            setMessage(`Analysis complete! Found ${response.data.analysis.anomaly_count} anomalies out of ${response.data.analysis.total_samples} samples.`);
            
            // Save the analysis to cookies with file name
            const dataWithFileName = {
                ...response.data,
                fileName: selectedFile.name
            };
            const recordId = saveRecord(dataWithFileName);
            
            if (recordId) {
                console.log('✅ Record saved successfully with ID:', recordId);
                setMessage(`Analysis complete! Found ${response.data.analysis.anomaly_count} anomalies. Record saved to history.`);
            } else {
                console.warn('⚠️ Failed to save record to cookies');
                setMessage(`Analysis complete! Found ${response.data.analysis.anomaly_count} anomalies. (Warning: Could not save to history)`);
            }
            
            onUploadSuccess(response.data);
            setSelectedFile(null);
        } catch (error) {
            setMessage(`Error: ${error.response?.data?.error || error.message}`);
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="file-upload-container">
            <h2>Upload File</h2>
            <div className="upload-area">
                <input
                    type="file"
                    onChange={handleFileSelect}
                    disabled={uploading}
                    id="file-input"
                />
                <label htmlFor="file-input" className="file-label">
                    {selectedFile ? selectedFile.name : 'Choose a file'}
                </label>
                <button
                    onClick={handleUpload}
                    disabled={!selectedFile || uploading}
                    className="upload-button"
                >
                    {uploading ? 'Uploading...' : 'Upload'}
                </button>
            </div>
            {message && (
                <div className={`message ${message.includes('Error') ? 'error' : 'success'}`}>
                    {message}
                </div>
            )}
        </div>
    );
};

export default FileUpload;
