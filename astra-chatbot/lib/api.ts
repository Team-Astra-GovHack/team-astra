import type { Dataset, QueryResult, SuggestedQuery } from "./types"

// Enhanced mock data with more realistic government datasets
const mockDatasets: Dataset[] = [
  {
    id: "gov-spending-2024",
    name: "Government Spending 2024",
    description:
      "Federal government expenditures and budget allocations for fiscal year 2024, including departmental budgets, program funding, and emergency appropriations",
    recordCount: 15420,
    lastUpdated: "2024-12-15",
  },
  {
    id: "vendor-payments",
    name: "Vendor Payments",
    description:
      "Comprehensive database of payments made to government contractors and service providers, including contract details, payment amounts, and vendor information",
    recordCount: 8932,
    lastUpdated: "2024-12-10",
  },
  {
    id: "public-contracts",
    name: "Public Contracts",
    description:
      "Active and completed government contracts with private entities, including procurement details, contract values, and performance metrics",
    recordCount: 3456,
    lastUpdated: "2024-12-08",
  },
  {
    id: "federal-grants",
    name: "Federal Grants",
    description:
      "Grant awards to states, localities, nonprofits, and research institutions, including grant amounts, recipients, and program areas",
    recordCount: 12789,
    lastUpdated: "2024-12-12",
  },
  {
    id: "employee-salaries",
    name: "Federal Employee Salaries",
    description:
      "Salary and compensation data for federal employees, organized by agency, position, and pay grade while maintaining privacy compliance",
    recordCount: 2156789,
    lastUpdated: "2024-12-01",
  },
]

// Dataset-specific suggested queries
const suggestedQueriesByDataset: Record<string, SuggestedQuery[]> = {
  "gov-spending-2024": [
    {
      id: "spending-1",
      text: "What are the top 5 spending categories this fiscal year?",
      category: "Budget Overview",
    },
    {
      id: "spending-2",
      text: "Show me departments with budget increases over 10% from last year",
      category: "Trend Analysis",
    },
    {
      id: "spending-3",
      text: "Find emergency appropriations and their purposes",
      category: "Financial Analysis",
    },
    {
      id: "spending-4",
      text: "Compare defense vs non-defense spending allocations",
      category: "Budget Overview",
    },
  ],
  "vendor-payments": [
    {
      id: "vendor-1",
      text: "Show me vendor payment outliers above $1M",
      category: "Financial Analysis",
    },
    {
      id: "vendor-2",
      text: "Which vendors received the most payments this year?",
      category: "Procurement Analysis",
    },
    {
      id: "vendor-3",
      text: "Find duplicate or suspicious payment patterns",
      category: "Financial Analysis",
    },
    {
      id: "vendor-4",
      text: "Show payment trends by vendor size (small business vs large)",
      category: "Trend Analysis",
    },
  ],
  "public-contracts": [
    {
      id: "contract-1",
      text: "Find contracts awarded without competitive bidding",
      category: "Procurement Analysis",
    },
    {
      id: "contract-2",
      text: "Show me the largest contracts by value this year",
      category: "Financial Analysis",
    },
    {
      id: "contract-3",
      text: "Which agencies have the most contract modifications?",
      category: "Procurement Analysis",
    },
    {
      id: "contract-4",
      text: "Compare contract performance ratings across agencies",
      category: "Trend Analysis",
    },
  ],
  "federal-grants": [
    {
      id: "grant-1",
      text: "What are the largest grant awards by program area?",
      category: "Financial Analysis",
    },
    {
      id: "grant-2",
      text: "Show grant distribution by state and territory",
      category: "Budget Overview",
    },
    {
      id: "grant-3",
      text: "Find research grants over $5M in STEM fields",
      category: "Financial Analysis",
    },
    {
      id: "grant-4",
      text: "Compare grant success rates by recipient type",
      category: "Trend Analysis",
    },
  ],
  "employee-salaries": [
    {
      id: "salary-1",
      text: "Show average salary ranges by agency and position level",
      category: "Budget Overview",
    },
    {
      id: "salary-2",
      text: "Find agencies with highest compensation growth rates",
      category: "Trend Analysis",
    },
    {
      id: "salary-3",
      text: "Compare federal vs private sector salary benchmarks",
      category: "Financial Analysis",
    },
    {
      id: "salary-4",
      text: "Show geographic pay differentials across regions",
      category: "Budget Overview",
    },
  ],
}

// Default suggested queries when no dataset is selected
const defaultSuggestedQueries: SuggestedQuery[] = [
  {
    id: "default-1",
    text: "Show me vendor payment outliers above $1M",
    category: "Financial Analysis",
  },
  {
    id: "default-2",
    text: "What are the top 5 spending categories this year?",
    category: "Budget Overview",
  },
  {
    id: "default-3",
    text: "Find contracts awarded without competitive bidding",
    category: "Procurement Analysis",
  },
  {
    id: "default-4",
    text: "Compare spending trends vs last fiscal year",
    category: "Trend Analysis",
  },
]

// Enhanced response generation based on query content
const generateContextualResponse = (query: string, datasetName: string): string => {
  const lowerQuery = query.toLowerCase()

  if (lowerQuery.includes("outlier") || lowerQuery.includes("anomal")) {
    return `Analysis of ${datasetName} reveals several significant outliers in the requested data range. Our statistical analysis identified 12 transactions that exceed normal patterns by more than 3 standard deviations. The largest outlier represents a $15.2M payment that appears to be a legitimate infrastructure contract but warrants additional review due to its size. These outliers account for approximately 23% of total transaction volume despite representing only 0.8% of transaction count.`
  }

  if (lowerQuery.includes("top") || lowerQuery.includes("largest") || lowerQuery.includes("highest")) {
    return `Based on comprehensive analysis of ${datasetName}, the top-ranking items show clear patterns in government spending priorities. The highest values are concentrated in defense and infrastructure sectors, with the top 5 entries accounting for 34% of total allocated funds. Notable trends include increased investment in cybersecurity initiatives and renewable energy projects, reflecting current policy priorities and strategic objectives.`
  }

  if (lowerQuery.includes("trend") || lowerQuery.includes("compare") || lowerQuery.includes("vs")) {
    return `Trend analysis of ${datasetName} shows significant year-over-year changes in spending patterns. Overall expenditures increased by 8.3% compared to the previous fiscal year, with the most substantial growth in technology and healthcare sectors. Emergency appropriations contributed to 15% of the increase, while baseline program funding grew by 4.2%. Regional distribution patterns remain consistent with historical allocations.`
  }

  if (lowerQuery.includes("contract") || lowerQuery.includes("procurement") || lowerQuery.includes("bidding")) {
    return `Procurement analysis of ${datasetName} indicates strong adherence to competitive bidding requirements, with 87% of contracts awarded through full and open competition. Small business participation reached 28% of total contract value, exceeding federal targets. Contract modifications averaged 2.1 per contract, within normal ranges. Three contracts require additional review due to sole-source justifications, but all appear to meet regulatory requirements.`
  }

  // Default response
  return `Based on analysis of ${datasetName}, our comprehensive review reveals important insights aligned with your query. The data demonstrates consistent patterns with established government reporting standards and provides actionable information for policy analysis. Key findings indicate strong correlation with historical trends while highlighting areas of significant change that merit further investigation. All results have been cross-validated against multiple data sources to ensure accuracy and reliability.`
}

// Mock API functions with enhanced realism
export const fetchDatasets = async (): Promise<Dataset[]> => {
  // Simulate network delay with some variability
  const delay = 600 + Math.random() * 400
  await new Promise((resolve) => setTimeout(resolve, delay))

  // Simulate occasional network issues (5% chance)
  if (Math.random() < 0.05) {
    throw new Error("Network timeout - please try again")
  }

  return mockDatasets
}

export const submitQuery = async (query: string, datasetId: string): Promise<QueryResult> => {
  // Simulate processing time based on query complexity
  const queryComplexity = query.split(" ").length
  const baseDelay = 1200
  const complexityDelay = Math.min(queryComplexity * 50, 800)
  const totalDelay = baseDelay + complexityDelay + Math.random() * 300

  await new Promise((resolve) => setTimeout(resolve, totalDelay))

  // Simulate occasional processing errors (3% chance)
  if (Math.random() < 0.03) {
    throw new Error("Query processing failed - please rephrase your question and try again")
  }

  const dataset = mockDatasets.find((d) => d.id === datasetId)
  const datasetName = dataset?.name || "selected dataset"

  // Generate trust score based on query characteristics
  let trustScore: "high" | "medium" | "low"
  let confidence: number

  if (query.length > 50 && query.includes("show") && !query.includes("all")) {
    trustScore = "high"
    confidence = 85 + Math.random() * 10
  } else if (query.length > 20) {
    trustScore = "medium"
    confidence = 70 + Math.random() * 15
  } else {
    trustScore = "low"
    confidence = 55 + Math.random() * 15
  }

  const mockResult: QueryResult = {
    id: `query-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
    query,
    answer: generateContextualResponse(query, datasetName),
    trustScore,
    confidence: Math.round(confidence * 100) / 100,
    sources: [
      `${datasetName} Database`,
      "Federal Financial Management System",
      "Treasury Payment Records",
      ...(trustScore === "high" ? ["GAO Audit Reports", "OMB Budget Documents"] : []),
    ],
    reasoning: [
      "Parsed natural language query and identified key data requirements",
      `Accessed ${datasetName} with ${dataset?.recordCount.toLocaleString()} records`,
      "Applied statistical analysis and filtering based on query parameters",
      "Cross-referenced results with historical data patterns for validation",
      ...(trustScore === "high"
        ? [
            "Verified findings against independent audit reports",
            "Confirmed compliance with federal reporting standards",
          ]
        : []),
      "Generated response with confidence scoring and source attribution",
    ],
    timestamp: new Date().toISOString(),
  }

  return mockResult
}

export const fetchSuggestedQueries = async (datasetId?: string): Promise<SuggestedQuery[]> => {
  await new Promise((resolve) => setTimeout(resolve, 200 + Math.random() * 200))

  if (datasetId && suggestedQueriesByDataset[datasetId]) {
    return suggestedQueriesByDataset[datasetId]
  }

  return defaultSuggestedQueries
}

// Additional utility functions for data management
export const validateQuery = (query: string): { isValid: boolean; suggestions?: string[] } => {
  const trimmedQuery = query.trim()

  if (trimmedQuery.length < 3) {
    return { isValid: false, suggestions: ["Please enter a more detailed question"] }
  }

  if (trimmedQuery.length > 500) {
    return { isValid: false, suggestions: ["Please shorten your query to under 500 characters"] }
  }

  // Check for potentially problematic queries
  const problematicTerms = ["password", "ssn", "social security", "personal", "private"]
  const hasProblematicTerms = problematicTerms.some((term) => trimmedQuery.toLowerCase().includes(term))

  if (hasProblematicTerms) {
    return {
      isValid: false,
      suggestions: [
        "This query may request sensitive information",
        "Please rephrase to focus on aggregate or public data",
      ],
    }
  }

  return { isValid: true }
}

export const getDatasetById = (datasetId: string): Dataset | undefined => {
  return mockDatasets.find((dataset) => dataset.id === datasetId)
}

export const searchDatasets = (searchTerm: string): Dataset[] => {
  const term = searchTerm.toLowerCase()
  return mockDatasets.filter(
    (dataset) => dataset.name.toLowerCase().includes(term) || dataset.description.toLowerCase().includes(term),
  )
}
