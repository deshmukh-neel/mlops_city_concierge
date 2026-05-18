/**
 * Props that make a non-<button> element behave like a button: clickable,
 * keyboard-operable (Enter/Space), focusable, labeled, with press feedback.
 * Single source of truth for the "div/span acting as a button" a11y contract.
 */
export function clickableProps(onClick, label) {
  return {
    role: 'button',
    tabIndex: 0,
    'aria-label': label,
    className: 'press',
    onClick,
    onKeyDown: (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault()
        onClick?.()
      }
    },
  }
}
