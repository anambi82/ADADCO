import React, { useState } from 'react';
import './App.css';
import FileUpload from './components/FileUpload';
import DataVisualization from './components/DataVisualization';

function App() {
  const [activeTab, setActiveTab] = useState('upload');
  const [uploadedData, setUploadedData] = useState(null);

  return (
    <div className="App">
      <header className="App-header">
        <h1>ADADCO Data Analysis</h1>
      </header>

      <div className="tabs">
        <button
          className={activeTab === 'upload' ? 'active' : ''}
          onClick={() => setActiveTab('upload')}
        >
          File Upload
        </button>
        <button
          className={activeTab === 'graphs' ? 'active' : ''}
          onClick={() => setActiveTab('graphs')}
        >
          Data Visualization
        </button>
      </div>

      <div className="content">
        {activeTab === 'upload' && (
          <FileUpload onUploadSuccess={setUploadedData} />
        )}
        {activeTab === 'graphs' && (
          <DataVisualization data={uploadedData} />
        )}
      </div>
    </div>
  );
}

export default App;
