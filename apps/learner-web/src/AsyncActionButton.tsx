import type { ButtonHTMLAttributes, ReactNode } from "react";

type AsyncActionButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  loading: boolean;
  loadingLabel: ReactNode;
};

/** A consistent, accessible affordance for mutations that may take noticeable time. */
export default function AsyncActionButton({
  loading,
  loadingLabel,
  children,
  disabled,
  ...props
}: AsyncActionButtonProps) {
  return (
    <button {...props} disabled={disabled || loading} aria-busy={loading}>
      {loading && <span className="async-button-spinner" aria-hidden="true" />}
      <span className="async-button-label">{loading ? loadingLabel : children}</span>
    </button>
  );
}
