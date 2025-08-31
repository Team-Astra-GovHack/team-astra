import { useMutation, useQueryClient } from "@tanstack/react-query"
import { submitQuery } from "@/lib/api"
import type { QueryResult } from "@/lib/types"

export function useQuerySubmission() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ query, datasetId }: { query: string; datasetId: string }) => submitQuery(query, datasetId),
    onSuccess: (data: QueryResult) => {
      // Cache the result for potential future reference
      queryClient.setQueryData(["query-result", data.id], data)

      // Invalidate and refetch any related queries if needed
      queryClient.invalidateQueries({ queryKey: ["query-history"] })
    },
    onError: (error) => {
      console.error("Query submission failed:", error)
    },
  })
}
