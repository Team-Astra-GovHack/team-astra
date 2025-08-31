"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft } from "lucide-react";
import Image from "next/image";
import Link from "next/link";

import { fetchDatasets, fetchSuggestedQueries } from "@/lib/api";
import type { QueryResult } from "@/lib/types";
import { buttonVariants } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { DatasetSelector } from "@/components/dataset-selector";
import { QueryInput } from "@/components/query-input";
import { ResponsePanel } from "@/components/response-panel";
import { SuggestedQuestions } from "@/components/suggested-questions";

export default function ChatbotPage() {
  const [selectedDataset, setSelectedDataset] = useState<string>("");
  const [queryHistory, setQueryHistory] = useState<QueryResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const { data: datasets, isLoading: datasetsLoading } = useQuery({
    queryKey: ["datasets"],
    queryFn: fetchDatasets,
  });

  const { data: suggestedQueries } = useQuery({
    queryKey: ["suggested-queries"],
    queryFn: () => fetchSuggestedQueries(),
  });

  const handleQuerySubmit = async (result: QueryResult) => {
    setQueryHistory((prev) => [result, ...prev]);
  };

  const handleSuggestedQuery = (queryText: string) => {
    const event = new CustomEvent("suggested-query", { detail: queryText });
    window.dispatchEvent(event);
  };

  return (
    // Use a flex layout to manage the overall page structure
    <div className="flex flex-col bg-background text-foreground p-4 sm:p-6 lg:p-8">
      <header className="flex-shrink-0">
        <Link href={"/"} className={buttonVariants({ variant: "secondary" })}>
          <ArrowLeft className="mr-2 h-4 w-4" /> Back
        </Link>
        <div className="flex justify-center my-4">
          <Image
            src="/logo1b.png"
            alt="ASTRA logo"
            width={100}
            height={100}
            className="rounded bg-transparent object-contain"
            priority
          />
        </div>
        <div className="w-full max-w-3xl mx-auto mb-8">
          <QueryInput
            selectedDataset={selectedDataset}
            onQuerySubmit={handleQuerySubmit}
            isLoading={isLoading}
            setIsLoading={setIsLoading}
          />
        </div>
      </header>

      {/* Main 3-column layout */}
      <main className="flex-grow w-full max-w-7xl mx-auto grid lg:grid-cols-4 gap-8 overflow-hidden">
        {/* Left Column (25%) */}
        <aside className="hidden lg:block lg:col-span-1 space-y-4 self-start sticky top-8">
          <h2 className="text-xl font-semibold">Select Dataset</h2>
          <Card className="bg-card/50">
            <CardContent className="p-4">
              <DatasetSelector
                datasets={datasets || []}
                selectedDataset={selectedDataset}
                onDatasetChange={setSelectedDataset}
                isLoading={datasetsLoading}
              />
            </CardContent>
          </Card>
        </aside>

        {/* Middle Column (50%) - Scrollable */}
        <div className="lg:col-span-2 h-full overflow-y-auto pr-4 space-y-4">
          {queryHistory.map((result, index) => (
            <ResponsePanel
              key={result.id}
              result={result}
              isLatest={index === 0}
            />
          ))}
          {queryHistory.length === 0 && !isLoading && (
            <Card className="border-dashed h-screen">
              <CardContent className="flex h-screen items-center justify-center">
                <div className="text-center space-y-2">
                  <p className="text-muted-foreground">
                    Your results will appear here
                  </p>
                  <p className="text-sm text-muted-foreground">
                    Select a dataset and ask a question to get started
                  </p>
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Right Column (25%) */}
        <aside className="hidden lg:block lg:col-span-1 space-y-8 self-start sticky top-8">
          <div>
            <h2 className="text-xl font-semibold mb-4">Suggested Qs</h2>
            <Card className="bg-card/50">
              <CardContent className="p-4">
                <SuggestedQuestions
                  queries={suggestedQueries || []}
                  onQuerySelect={handleSuggestedQuery}
                  disabled={!selectedDataset}
                />
              </CardContent>
            </Card>
          </div>
          <div>
            <h2 className="text-xl font-semibold mb-4">Tips</h2>
            <Card className="bg-card/50">
              <CardContent className="p-4 space-y-3">
                <div className="text-sm space-y-1">
                  <p className="font-medium text-foreground">Be specific:</p>
                  <p className="text-muted-foreground">
                    Include time periods, amounts, or categories.
                  </p>
                </div>
                <Separator />
                <div className="text-sm space-y-1">
                  <p className="font-medium text-foreground">
                    Use natural language:
                  </p>
                  <p className="text-muted-foreground">
                    Ask questions as you would to a human.
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        </aside>
      </main>
    </div>
  );
}
