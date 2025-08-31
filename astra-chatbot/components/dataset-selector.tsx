"use client"

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Database, Calendar, Hash } from "lucide-react"
import { DatasetSelectorSkeleton } from "@/components/loading-states"
import { ScreenReaderOnly } from "@/components/accessibility/screen-reader-only"
import type { Dataset } from "@/lib/types"

interface DatasetSelectorProps {
  datasets: Dataset[]
  selectedDataset: string
  onDatasetChange: (datasetId: string) => void
  isLoading: boolean
}

export function DatasetSelector({ datasets, selectedDataset, onDatasetChange, isLoading }: DatasetSelectorProps) {
  if (isLoading) {
    return (
      <div role="status" aria-label="Loading datasets">
        <ScreenReaderOnly>Loading available datasets...</ScreenReaderOnly>
        <DatasetSelectorSkeleton />
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Dropdown Selector */}
      <div>
        <label htmlFor="dataset-select" className="sr-only">
          Select a dataset to query
        </label>
        <Select value={selectedDataset} onValueChange={onDatasetChange}>
          <SelectTrigger className="w-full" id="dataset-select" aria-describedby="dataset-help">
            <SelectValue placeholder="Choose a dataset to query..." />
          </SelectTrigger>
          <SelectContent>
            {datasets.map((dataset) => (
              <SelectItem key={dataset.id} value={dataset.id}>
                <div className="flex items-center space-x-2">
                  <Database className="h-4 w-4" aria-hidden="true" />
                  <span>{dataset.name}</span>
                </div>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <p id="dataset-help" className="sr-only">
          Select a government dataset to query. You can also click on the cards below to make a selection.
        </p>
      </div>

      {/* Dataset Cards */}
      <div className="grid gap-3" role="radiogroup" aria-labelledby="dataset-cards-label">
        <h3 id="dataset-cards-label" className="sr-only">
          Available Datasets
        </h3>
        {datasets.map((dataset) => (
          <Card
            key={dataset.id}
            className={`cursor-pointer transition-all hover:shadow-md focus-within:ring-2 focus-within:ring-accent ${
              selectedDataset === dataset.id ? "ring-2 ring-accent border-accent bg-accent/5" : "hover:border-accent/50"
            }`}
            role="radio"
            aria-checked={selectedDataset === dataset.id}
            tabIndex={0}
            onClick={() => onDatasetChange(dataset.id)}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault()
                onDatasetChange(dataset.id)
              }
            }}
            aria-describedby={`dataset-${dataset.id}-description`}
          >
            <CardContent className="p-4">
              <div className="flex items-start justify-between">
                <div className="space-y-2 flex-1">
                  <div className="flex items-center space-x-2">
                    <Database className="h-4 w-4 text-accent" aria-hidden="true" />
                    <h3 className="font-medium text-foreground">{dataset.name}</h3>
                  </div>
                  <p id={`dataset-${dataset.id}-description`} className="text-sm text-muted-foreground text-pretty">
                    {dataset.description}
                  </p>
                  <div className="flex items-center space-x-4 text-xs text-muted-foreground">
                    <div className="flex items-center space-x-1">
                      <Hash className="h-3 w-3" aria-hidden="true" />
                      <span>
                        <ScreenReaderOnly>Contains </ScreenReaderOnly>
                        {dataset.recordCount.toLocaleString()} records
                      </span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <Calendar className="h-3 w-3" aria-hidden="true" />
                      <span>
                        <ScreenReaderOnly>Last updated </ScreenReaderOnly>
                        Updated {new Date(dataset.lastUpdated).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                </div>
                {selectedDataset === dataset.id && (
                  <Badge variant="secondary" className="ml-2" aria-label="Currently selected dataset">
                    Selected
                  </Badge>
                )}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}
