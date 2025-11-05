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

// Save a new record to cookies - MINIMAL VERSION (only what's displayed)
export const saveRecord = (analysisData) => {
    try {
        console.log('Saving record for:', analysisData.fileName || analysisData.filename);
        const records = getRecords();
        console.log('Current records count:', records.length);
        
        // Extract ONLY the top 3 attack types (that's all we show in the UI)
        const top3Attacks = analysisData.attack_breakdown 
            ? Object.entries(analysisData.attack_breakdown)
                .map(([attack_type, data]) => ({
                    attack_type,
                    count: data.count
                }))
                .sort((a, b) => b.count - a.count)
                .slice(0, 3) // Only top 3 for display
            : [];
        
        // Create MINIMAL record - only store what's displayed in Records tab
        const newRecord = {
            id: Date.now(),
            timestamp: new Date().toISOString(),
            fileName: analysisData.fileName || analysisData.filename || 'Unknown File',
            // Minimal summary - only displayed fields
            summary: {
                total_samples: analysisData.analysis?.total_samples || 0,
                benign_count: analysisData.analysis?.benign_count || 0,
                anomaly_count: analysisData.analysis?.anomaly_count || 0,
                anomaly_rate: analysisData.analysis?.anomaly_rate || 0,
                severity: analysisData.summary?.severity || 'LOW'
            },
            // Only top 3 attacks for the card display
            attacks: top3Attacks,
            // Total attack count for "+X more" display
            totalAttackTypes: analysisData.summary?.unique_attack_types || top3Attacks.length
        };
        
        // Add new record at the beginning
        records.unshift(newRecord);
        
        // Keep only the most recent records
        let recordsToSave = records.slice(0, MAX_RECORDS);
        
        // Try to fit records in cookie, removing oldest ones if needed
        const MAX_COOKIE_SIZE = 3500; // Leave buffer below 4KB limit
        let cookieData = JSON.stringify(recordsToSave);
        let cookieSize = encodeURIComponent(cookieData).length;
        
        console.log('Initial cookie size:', cookieSize, 'bytes with', recordsToSave.length, 'records');
        
        // Keep removing the oldest record until it fits
        while (cookieSize > MAX_COOKIE_SIZE && recordsToSave.length > 1) {
            const removedRecord = recordsToSave.pop(); // Remove oldest (last) record
            console.log('🗑️ Removing oldest record to make space:', removedRecord.fileName);
            
            cookieData = JSON.stringify(recordsToSave);
            cookieSize = encodeURIComponent(cookieData).length;
            console.log('New cookie size:', cookieSize, 'bytes with', recordsToSave.length, 'records');
        }
        
        // Final check - if still too large with just the new record, something is wrong
        if (cookieSize > MAX_COOKIE_SIZE && recordsToSave.length === 1) {
            console.error('❌ Single record is too large for cookie storage!');
            console.error('Record size:', cookieSize, 'bytes. This should not happen with minimal format.');
            return null;
        }
        
        // Save to cookie (expires in 30 days)
        const expires = new Date();
        expires.setDate(expires.getDate() + 30);
        
        document.cookie = `${COOKIE_NAME}=${encodeURIComponent(cookieData)}; expires=${expires.toUTCString()}; path=/`;
        
        console.log('✅ Record saved successfully! Total records:', recordsToSave.length, '| Cookie size:', cookieSize, 'bytes');
        return newRecord.id;
    } catch (error) {
        console.error('❌ Error saving record to cookies:', error);
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

