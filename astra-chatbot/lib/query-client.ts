import { QueryClient } from "@tanstack/react-query"

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      gcTime: 10 * 60 * 1000, // 10 minutes (formerly cacheTime)
      retry: (failureCount, error) => {
        // Don't retry on validation errors
        if (error instanceof Error && error.message.includes("validation")) {
          return false
        }
        // Retry up to 2 times for network errors
        return failureCount < 2
      },
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
      refetchOnWindowFocus: false,
      refetchOnReconnect: true,
    },
    mutations: {
      retry: 1,
      retryDelay: 1000,
    },
  },
})

// Global error handler for queries
queryClient.setMutationDefaults(["submit-query"], {
  mutationFn: async ({ query, datasetId }: { query: string; datasetId: string }) => {
    const { submitQuery } = await import("@/lib/api")
    return submitQuery(query, datasetId)
  },
})
