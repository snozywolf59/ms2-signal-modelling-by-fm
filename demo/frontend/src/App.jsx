import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const PeptideDemo = () => {
  const [peptide, setPeptide] = useState('PEPTIDEK');
  const [charge, setCharge] = useState(2);
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);

  // Giả lập hàm gọi API từ Model AI/Bioinformatics
  const handlePredict = () => {
    setLoading(true);

    // Giả lập delay của model
    setTimeout(() => {
      const mockData = generateMockData(peptide);
      setData(mockData);
      setLoading(false);
    }, 800);
  };

  // Hàm tạo dữ liệu demo (Thực tế sẽ nhận từ Backend)
  const generateMockData = (seq) => {
    return seq.split('').map((amino, index) => ({
      name: `${index + 1}`,
      mz: Math.floor(Math.random() * 1000) + 100,
      intensity: Math.random() * 100,
      ionType: Math.random() > 0.5 ? 'b' : 'y'
    }));
  };

  return (
    <div className="flex h-screen bg-gray-900 text-white font-sans">
      {/* Sidebar - Control Panel */}
      <div className="w-80 bg-gray-800 p-6 border-r border-gray-700 flex flex-col gap-6">
        <h1 className="text-xl font-bold border-b border-gray-700 pb-4 text-blue-400">
          Peptide Fragment Predictor
        </h1>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2 text-gray-300">Peptide Sequence</label>
            <input
              type="text"
              value={peptide}
              onChange={(e) => setPeptide(e.target.value.toUpperCase())}
              className="w-full p-2.5 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none uppercase tracking-widest"
              placeholder="e.g. PEPTIDEK"
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-2 text-gray-300">Precursor Charge</label>
            <input
              type="number"
              value={charge}
              onChange={(e) => setCharge(e.target.value)}
              className="w-full p-2.5 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
              min="1" max="6"
            />
          </div>

          <button
            onClick={handlePredict}
            disabled={loading}
            className={`w-full py-3 rounded-lg font-semibold transition-all ${
              loading ? 'bg-gray-600' : 'bg-blue-600 hover:bg-blue-500 active:scale-95'
            }`}
          >
            {loading ? 'Predicting...' : 'Predict Intensity'}
          </button>
        </div>

        <div className="mt-auto text-xs text-gray-500 italic">
          * Model version: v1.0.4-beta <br/>
          * Support b/y fragment ions
        </div>
      </div>

      {/* Main Content - Visualization */}
      <div className="flex-1 p-8 flex flex-col">
        <div className="bg-gray-800 rounded-xl p-6 shadow-2xl border border-gray-700 flex-1 flex flex-col">
          <div className="flex justify-between items-center mb-8">
            <h2 className="text-lg font-semibold italic text-gray-400">Predicted Fragment Intensity Map</h2>
            <div className="flex gap-4">
              <span className="flex items-center gap-2 text-sm"><span className="w-3 h-3 bg-red-500 rounded-full"></span> b-ions</span>
              <span className="flex items-center gap-2 text-sm"><span className="w-3 h-3 bg-blue-500 rounded-full"></span> y-ions</span>
            </div>
          </div>

          <div className="flex-1 min-h-[400px]">
            {data.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
                  <XAxis dataKey="mz" label={{ value: 'm/z', position: 'insideBottom', offset: -10, fill: '#9CA3AF' }} />
                  <YAxis label={{ value: 'Intensity', angle: -90, position: 'insideLeft', fill: '#9CA3AF' }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '8px' }}
                    itemStyle={{ color: '#F3F4F6' }}
                  />
                  <Bar dataKey="intensity">
                    {data.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.ionType === 'b' ? '#EF4444' : '#3B82F6'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-gray-500 border-2 border-dashed border-gray-700 rounded-lg">
                Enter peptide sequence and run prediction to see results
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default PeptideDemo;
