// StockCategory.tsx
import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown } from 'lucide-react';

type StockCategoryProps = {
  title: string;
  stocks: { symbol: string; name: string }[];
  onSelectStock: (symbol: string) => void;
};

const StockCategory: React.FC<StockCategoryProps> = ({ title, stocks, onSelectStock }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="mb-2 rounded-md bg-gray-800 overflow-hidden">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex justify-between items-center p-4 text-left text-white"
      >
        <span className="text-lg font-medium">{title}</span>
        <ChevronDown 
          className={`transform transition-transform ${isOpen ? 'rotate-180' : ''}`} 
          size={20} 
        />
      </button>
      
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0 }}
            animate={{ height: 'auto' }}
            exit={{ height: 0 }}
            transition={{ duration: 0.3 }}
            className="overflow-hidden"
          >
            <div className="p-2 bg-gray-900">
              {stocks.map((stock) => (
                <button
                  key={stock.symbol}
                  onClick={() => onSelectStock(stock.symbol)}
                  className="w-full p-2 text-left text-white hover:bg-gray-700 rounded"
                >
                  {stock.symbol}: {stock.name}
                </button>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default StockCategory;