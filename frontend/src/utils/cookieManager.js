// Utility for managing analysis records in cookies

const COOKIE_NAME = 'analysis_records';
const MAX_RECORDS = 10; // Limit to prevent cookie size issues

// Get all records from cookies
export const getRecords = () => {
    try {
        const cookie = document.cookie
            .split('; ')
            .find(row => row.startsWith(`${COOKIE_NAME}=`));
        
        if (!cookie) return [];
        
        const value = cookie.split('=')[1];
        return JSON.parse(decodeURIComponent(value));
    } catch (error) {
        console.error('Error reading records from cookies:', error);
        return [];
    }
};

// Save a new record to cookies
export const saveRecord = (analysisData) => {
    try {
        const records = getRecords();
        
        // Create a new record with timestamp and ID
        // Combine analysis data with summary severity info
        const combinedSummary = {
            ...(analysisData.analysis || {}),
            severity: analysisData.summary?.severity,
            unique_attack_types: analysisData.summary?.unique_attack_types,
            most_common_attack: analysisData.summary?.most_common_attack
        };
        
        // Convert attack_breakdown to attacks array format
        const attacksArray = analysisData.attack_breakdown 
            ? Object.entries(analysisData.attack_breakdown).map(([attack_type, data]) => ({
                attack_type,
                count: data.count,
                percentage: data.percentage
            }))
            : (analysisData.attacks || []);
        
        // Format confidence data
        const confidenceData = analysisData.attack_confidence 
            ? {
                confidence_by_attack: Object.entries(analysisData.attack_confidence).map(([attack_type, avg_confidence]) => ({
                    attack_type,
                    average_confidence: avg_confidence
                })),
                statistics: null // Will be calculated if needed
            }
            : (analysisData.confidence || null);
        
        const newRecord = {
            id: Date.now(),
            timestamp: new Date().toISOString(),
            fileName: analysisData.fileName || analysisData.filename || 'Unknown File',
            summary: combinedSummary,
            attacks: attacksArray,
            confidence: confidenceData
        };
        
        // Add new record at the beginning
        records.unshift(newRecord);
        
        // Keep only the most recent records
        const limitedRecords = records.slice(0, MAX_RECORDS);
        
        // Save to cookie (expires in 30 days)
        const expires = new Date();
        expires.setDate(expires.getDate() + 30);
        
        document.cookie = `${COOKIE_NAME}=${encodeURIComponent(JSON.stringify(limitedRecords))}; expires=${expires.toUTCString()}; path=/`;
        
        return newRecord.id;
    } catch (error) {
        console.error('Error saving record to cookies:', error);
        return null;
    }
};

// Get a specific record by ID
export const getRecordById = (id) => {
    const records = getRecords();
    return records.find(record => record.id === id);
};

// Delete a record by ID
export const deleteRecord = (id) => {
    try {
        const records = getRecords();
        const filteredRecords = records.filter(record => record.id !== id);
        
        const expires = new Date();
        expires.setDate(expires.getDate() + 30);
        
        document.cookie = `${COOKIE_NAME}=${encodeURIComponent(JSON.stringify(filteredRecords))}; expires=${expires.toUTCString()}; path=/`;
        
        return true;
    } catch (error) {
        console.error('Error deleting record from cookies:', error);
        return false;
    }
};

// Clear all records
export const clearAllRecords = () => {
    try {
        document.cookie = `${COOKIE_NAME}=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/`;
        return true;
    } catch (error) {
        console.error('Error clearing records from cookies:', error);
        return false;
    }
};

