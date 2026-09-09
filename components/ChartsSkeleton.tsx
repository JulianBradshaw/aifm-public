import { HeaderContent } from './HeaderContent';

// Skeleton shown during PPR streaming when custom params are being loaded
// This is part of the static prerender shell
export function ChartsSkeleton() {
  return (
    <div className="grid h-screen w-full grid-cols-[minmax(0,1fr)_auto] grid-rows-[minmax(0,1fr)] gap-0">
      <div className="relative col-start-1 row-start-1 flex min-h-0 flex-col">
        <div className="flex min-h-0 flex-col overflow-y-auto pb-10">
          <HeaderContent variant="inline" className="pt-6 pb-4 pl-6 lg:pl-8 pr-6" />
          <main className="w-full px-0 lg:px-8">
            <div className="flex flex-col flex-1 min-h-0">
              {/* Chart skeleton */}
              <div className="grid gap-4" style={{ gridTemplateColumns: '300px 1fr' }}>
                {/* Left panel skeleton */}
                <div className="space-y-4 p-4">
                  <div className="animate-pulse space-y-3">
                    <div className="h-4 bg-gray-200 rounded w-3/4" />
                    <div className="h-4 bg-gray-200 rounded w-full" />
                    <div className="h-4 bg-gray-200 rounded w-5/6" />
                  </div>
                  <div className="space-y-2 pt-4">
                    <div className="h-12 bg-gray-200 rounded animate-pulse" />
                    <div className="h-12 bg-gray-200 rounded animate-pulse" />
                  </div>
                </div>
                {/* Charts skeleton */}
                <div className="grid grid-cols-3 gap-4">
                  <div className="h-[300px] bg-gray-100 rounded animate-pulse flex items-center justify-center">
                    <div className="w-8 h-8 border-2 border-gray-300 border-t-gray-500 rounded-full animate-spin" />
                  </div>
                  <div className="h-[300px] bg-gray-100 rounded animate-pulse flex items-center justify-center">
                    <div className="w-8 h-8 border-2 border-gray-300 border-t-gray-500 rounded-full animate-spin" />
                  </div>
                  <div className="h-[300px] bg-gray-100 rounded animate-pulse flex items-center justify-center">
                    <div className="w-8 h-8 border-2 border-gray-300 border-t-gray-500 rounded-full animate-spin" />
                  </div>
                </div>
              </div>
              {/* Parameters skeleton */}
              <div className="mt-4 grid grid-cols-5 gap-4">
                {[1, 2, 3, 4, 5].map(i => (
                  <div key={i} className="space-y-2">
                    <div className="h-3 bg-gray-200 rounded w-full animate-pulse" />
                    <div className="h-6 bg-gray-200 rounded animate-pulse" />
                  </div>
                ))}
              </div>
            </div>
          </main>
        </div>
      </div>
    </div>
  );
}

