import React, { useState } from 'react';
import { Upload, BarChart3 } from 'lucide-react';

export default function FeatherApp() {
  const [selectedPipeline, setSelectedPipeline] = useState('Pipeline One');
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);

  const pipelines = ['Pipeline One', 'Pipeline Two', 'Pipeline Three'];

  const handleUpload = () => {
    console.log('Upload video clicked');
    // Add upload logic here
  };

  const handlePerformanceEval = () => {
    console.log('Performance evaluation clicked');
    // Add performance evaluation logic here
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-cyan-50 via-teal-50 to-blue-100 flex flex-col relative overflow-hidden">
      {/* Background decorative elements */}
      <div className="absolute top-20 left-10 w-32 h-32 bg-cyan-200 rounded-full opacity-20 blur-xl"></div>
      <div className="absolute bottom-40 right-20 w-40 h-40 bg-teal-200 rounded-full opacity-20 blur-xl"></div>
      <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-aqua-100 rounded-full opacity-10 blur-2xl"></div>

      {/* Top Center - Title */}
      <div className="flex justify-center pt-12 pb-8">
        <h1 className="text-6xl md:text-7xl lg:text-8xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-600 via-teal-600 to-blue-600 tracking-tight">
          FEATHER
        </h1>
      </div>

      {/* Middle - Upload Button */}
      <div className="flex-1 flex items-center justify-center px-4">
        <button
          onClick={handleUpload}
          className="group relative px-12 py-8 text-xl font-semibold text-cyan-700 bg-white/20 backdrop-blur-md border border-white/30 rounded-2xl shadow-lg hover:shadow-xl hover:bg-white/30 transition-all duration-300 transform hover:scale-105 active:scale-95"
        >
          <div className="flex items-center space-x-4">
            <Upload className="w-8 h-8 group-hover:rotate-12 transition-transform duration-300" />
            <span>Upload Video</span>
          </div>

          {/* Animated border effect */}
          <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-cyan-400 via-teal-400 to-blue-400 opacity-0 group-hover:opacity-20 transition-opacity duration-300"></div>
        </button>
      </div>

      {/* Bottom Left - Pipeline Dropdown */}
      <div className="absolute bottom-8 left-8">
        <div className="relative">
          <label className="block text-sm font-medium text-cyan-700 mb-2">Pipeline</label>
          <button
            onClick={() => setIsDropdownOpen(!isDropdownOpen)}
            className="flex items-center justify-between w-48 px-4 py-3 text-cyan-700 bg-white/20 backdrop-blur-md border border-white/30 rounded-xl shadow-lg hover:shadow-xl hover:bg-white/30 transition-all duration-300"
          >
            <span className="font-medium">{selectedPipeline}</span>
            <svg
              className={`w-5 h-5 transform transition-transform duration-200 ${
                isDropdownOpen ? 'rotate-180' : ''
              }`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          {/* Dropdown Menu */}
          {isDropdownOpen && (
            <div className="absolute bottom-full mb-2 w-48 bg-white/20 backdrop-blur-md border border-white/30 rounded-xl shadow-lg overflow-hidden z-10">
              {pipelines.map((pipeline, index) => (
                <button
                  key={index}
                  onClick={() => {
                    setSelectedPipeline(pipeline);
                    setIsDropdownOpen(false);
                  }}
                  className={`w-full px-4 py-3 text-left font-medium transition-all duration-200 ${
                    selectedPipeline === pipeline
                      ? 'bg-cyan-200/30 text-cyan-800'
                      : 'text-cyan-700 hover:bg-white/20'
                  }`}
                >
                  {pipeline}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Bottom Middle - Slogan */}
      <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2">
        <p className="text-lg font-medium text-cyan-700/80 italic tracking-wide">
          train light, row right
        </p>
      </div>

      {/* Bottom Right - Performance Evaluation Button */}
      <div className="absolute bottom-8 right-8">
        <button
          onClick={handlePerformanceEval}
          className="group flex items-center space-x-3 px-6 py-3 text-cyan-700 font-semibold bg-white/20 backdrop-blur-md border border-white/30 rounded-xl shadow-lg hover:shadow-xl hover:bg-white/30 transition-all duration-300 transform hover:scale-105 active:scale-95"
        >
          <BarChart3 className="w-5 h-5 group-hover:scale-110 transition-transform duration-300" />
          <span>Performance Evaluation</span>
        </button>
      </div>

      {/* Click outside to close dropdown */}
      {isDropdownOpen && (
        <div
          className="fixed inset-0 z-0"
          onClick={() => setIsDropdownOpen(false)}
        />
      )}
    </div>
  );
}