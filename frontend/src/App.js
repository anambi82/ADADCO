import React, { useState } from 'react';
import './App.css';
import FileUpload from './components/FileUpload';
import DataVisualization from './components/DataVisualization';
import Records from './components/Records';

function App() {
  const [activeTab, setActiveTab] = useState('upload');
  const [uploadedData, setUploadedData] = useState(null);
  const [selectedRecord, setSelectedRecord] = useState(null);

  const handleRecordSelect = (record) => {
    setSelectedRecord(record);
    if (record) {
      setActiveTab('graphs');
    }
  };

  const handleTabChange = (tab) => {
    setActiveTab(tab);
    if (tab !== 'graphs') {
      setSelectedRecord(null);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>ADADCO Data Analysis</h1>
      </header>

      <div className="tabs">
        <button
          className={activeTab === 'upload' ? 'active' : ''}
          onClick={() => handleTabChange('upload')}
        >
          File Upload
        </button>
        <button
          className={activeTab === 'graphs' ? 'active' : ''}
          onClick={() => handleTabChange('graphs')}
        >
          Data Visualization
        </button>
        <button
          className={activeTab === 'records' ? 'active' : ''}
          onClick={() => handleTabChange('records')}
        >
          Records
        </button>
      </div>

      <div className="content">
        {activeTab === 'upload' && (
          <FileUpload onUploadSuccess={setUploadedData} />
        )}
        {activeTab === 'graphs' && (
          <DataVisualization 
            data={uploadedData} 
            selectedRecord={selectedRecord}
          />
        )}
        {activeTab === 'records' && (
          <Records onSelectRecord={handleRecordSelect} />
        )}
      </div>
    </div>
  );
}

export default App;
