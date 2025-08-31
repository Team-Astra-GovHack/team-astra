<<<<<<< HEAD
"use client";

import { useTheme } from "next-themes";
import { Toaster as Sonner, ToasterProps } from "sonner";

const Toaster = ({ ...props }: ToasterProps) => {
  // Always use dark theme
  return (
    <Sonner
      theme="dark"
=======
"use client"

import { useTheme } from "next-themes"
import { Toaster as Sonner, ToasterProps } from "sonner"

const Toaster = ({ ...props }: ToasterProps) => {
  const { theme = "system" } = useTheme()

  return (
    <Sonner
      theme={theme as ToasterProps["theme"]}
>>>>>>> a77c106c79235fef58d739aec75bf69204f9ec0b
      className="toaster group"
      style={
        {
          "--normal-bg": "var(--popover)",
          "--normal-text": "var(--popover-foreground)",
          "--normal-border": "var(--border)",
        } as React.CSSProperties
      }
      {...props}
    />
<<<<<<< HEAD
  );
};

export { Toaster };
=======
  )
}

export { Toaster }
>>>>>>> a77c106c79235fef58d739aec75bf69204f9ec0b
