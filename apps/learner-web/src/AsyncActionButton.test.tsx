import { render, screen } from "@testing-library/react";
import { describe, expect, test } from "vitest";

import AsyncActionButton from "./AsyncActionButton";

describe("AsyncActionButton", () => {
  test("announces and disables an action while it is pending", () => {
    const { container } = render(
      <AsyncActionButton loading loadingLabel="Launching…">
        Launch
      </AsyncActionButton>,
    );

    const button = screen.getByRole("button", { name: "Launching…" });
    expect(button).toBeDisabled();
    expect(button).toHaveAttribute("aria-busy", "true");
    expect(container.querySelector(".async-button-spinner")).toBeInTheDocument();
  });

  test("preserves the normal label when idle", () => {
    render(
      <AsyncActionButton loading={false} loadingLabel="Saving…">
        Save
      </AsyncActionButton>,
    );

    expect(screen.getByRole("button", { name: "Save" })).toBeEnabled();
  });
});
