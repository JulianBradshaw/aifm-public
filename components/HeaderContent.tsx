import Link from "next/link";

interface HeaderContentProps {
  variant: 'inline' | 'floating';
  className?: string;
  onOpenAdvancedParams?: () => void;
}

export const HeaderContent = ({ variant, className = '', onOpenAdvancedParams }: HeaderContentProps) => (
  <header className={`flex flex-row items-start sm:items-center justify-between text-left gap-2 sm:gap-4 ${className}`}>
    <div className="flex flex-col sm:flex-row sm:items-center sm:flex-1 gap-2 sm:gap-4 min-w-0">
      <h1 className={`text-lg sm:text-2xl md:text-3xl font-bold text-primary font-et-book text-left leading-tight ${variant === 'inline' ? 'mb-0 sm:mb-2' : ''}`}>
        <Link href="/">AI Futures Model: Timelines & Takeoff</Link>
      </h1>
      <nav className="flex flex-row items-center gap-3 sm:gap-6 md:gap-10 text-sm flex-wrap sm:ml-auto sm:mr-10">
        <Link href="/about" className="font-system-mono text-xs whitespace-nowrap">
          About
        </Link>
        <Link href="/analysis" className="font-system-mono text-xs whitespace-nowrap">
          Analysis
        </Link>
        <Link href="/forecast" className="font-system-mono text-xs whitespace-nowrap px-2 py-1 rounded-md border border-black">
          Forecasts
        </Link>
      </nav>
    </div>
    {onOpenAdvancedParams && (
      <button
        type="button"
        onClick={onOpenAdvancedParams}
        className="sm:hidden p-2 -mt-1 -mr-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-md transition-colors flex-shrink-0"
        aria-label="Open advanced parameters"
      >
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 6h9.75M10.5 6a1.5 1.5 0 11-3 0m3 0a1.5 1.5 0 10-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-9.75 0h9.75" />
        </svg>
      </button>
    )}
  </header>
);
