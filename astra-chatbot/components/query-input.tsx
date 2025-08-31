// components/query-input.tsx

"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { Loader2, Send } from "lucide-react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { useEffect } from "react";

import { submitQuery } from "@/lib/api";
import type { QueryResult } from "@/lib/types";
import { Button } from "@/components/ui/button";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormMessage,
} from "@/components/ui/form";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/components/ui/use-toast";

const formSchema = z.object({
  query: z.string().min(5, {
    message: "Query must be at least 5 characters.",
  }),
});

interface QueryInputProps {
  selectedDataset: string;
  onQuerySubmit: (result: QueryResult) => void;
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
}

export function QueryInput({
  selectedDataset,
  onQuerySubmit,
  isLoading,
  setIsLoading,
}: QueryInputProps) {
  const { toast } = useToast();

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      query: "",
    },
  });

  const onSubmit = async (data: z.infer<typeof formSchema>) => {
    if (!selectedDataset) {
      toast({
        variant: "destructive",
        title: "No Dataset Selected",
        description: "Please select a dataset before asking a question.",
      });
      return;
    }

    setIsLoading(true);
    try {
      const result = await submitQuery(data.query, selectedDataset);
      onQuerySubmit(result);
    } catch (error) {
      console.error("Query submission failed:", error);
      // Create an error result object to display in the UI; cast to QueryResult to satisfy TypeScript
      const errorResult = {
        id: crypto.randomUUID(),
        query: data.query,
        message:
          error instanceof Error
            ? error.message
            : "An unknown error occurred. Please try again.",
      } as unknown as QueryResult;
      onQuerySubmit(errorResult);
    } finally {
      setIsLoading(false);
      form.reset();
    }
  };

  // Effect to listen for suggested query events
  useEffect(() => {
    const handleSuggestedQuery = (event: CustomEvent<string>) => {
      form.setValue("query", event.detail);
    };
    window.addEventListener(
      "suggested-query",
      handleSuggestedQuery as EventListener
    );
    return () => {
      window.removeEventListener(
        "suggested-query",
        handleSuggestedQuery as EventListener
      );
    };
  }, [form]);

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
        <FormField
          control={form.control}
          name="query"
          render={({ field }) => (
            <FormItem>
              <FormControl>
                <Textarea
                  placeholder="e.g., What was the total expenditure in the last quarter?"
                  className="resize-none"
                  {...field}
                />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        <Button
          type="submit"
          disabled={isLoading || !selectedDataset}
          className="w-full"
        >
          {isLoading ? (
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          ) : (
            <Send className="mr-2 h-4 w-4" />
          )}
          Ask Question
        </Button>
      </form>
    </Form>
  );
}
