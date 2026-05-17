// Vitest global setup: extends expect() with jest-dom matchers and
// provides a default fetch mock so adapter tests never hit the network.
import '@testing-library/jest-dom/vitest'
import { afterEach, vi } from 'vitest'

afterEach(() => {
  vi.restoreAllMocks()
})
