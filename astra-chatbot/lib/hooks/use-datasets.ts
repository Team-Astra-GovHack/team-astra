import { useQuery } from "@tanstack/react-query"
import { fetchDatasets } from "@/lib/api"

export function useDatasets() {
  return useQuery({
    queryKey: ["datasets"],
    queryFn: fetchDatasets,
    staleTime: 10 * 60 * 1000, // 10 minutes - datasets don't change often
    gcTime: 30 * 60 * 1000, // 30 minutes
  })
}
