import { useQuery } from "@tanstack/react-query"
import { fetchSuggestedQueries } from "@/lib/api"

export function useSuggestedQueries(datasetId?: string) {
  return useQuery({
    queryKey: ["suggested-queries", datasetId],
    queryFn: () => fetchSuggestedQueries(datasetId),
    enabled: !!datasetId, // Only fetch when dataset is selected
    staleTime: 5 * 60 * 1000, // 5 minutes
  })
}
