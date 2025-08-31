import { z } from "zod"

export interface Dataset {
  id: string
  name: string
  description: string
  recordCount: number
  lastUpdated: string
}

export interface QueryResult {
  id: string
  query: string
  answer: string
  trustScore: "high" | "medium" | "low"
  confidence: number
  sources: string[]
  reasoning: string[]
  timestamp: string
}

export interface SuggestedQuery {
  id: string
  text: string
  category: string
}

// Zod validation schemas
export const queryInputSchema = z.object({
  query: z
    .string()
    .min(3, "Query must be at least 3 characters long")
    .max(500, "Query must be less than 500 characters")
    .refine((val) => val.trim().length > 0, "Query cannot be empty or only whitespace"),
  datasetId: z.string().min(1, "Please select a dataset"),
})

export type QueryInput = z.infer<typeof queryInputSchema>
