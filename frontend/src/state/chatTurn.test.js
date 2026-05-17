import { describe, expect, it } from 'vitest'
import { buildHistory, assistantContentFor } from './chatTurn'

describe('buildHistory', () => {
  it('maps prior messages to {role, content}', () => {
    const msgs = [
      { role: 'assistant', content: 'hi' },
      { role: 'user', content: 'tacos?' },
    ]
    expect(buildHistory(msgs)).toEqual([
      { role: 'assistant', content: 'hi' },
      { role: 'user', content: 'tacos?' },
    ])
  })

  it('coerces non-string content to empty string', () => {
    const msgs = [{ role: 'assistant', content: { jsx: true } }]
    expect(buildHistory(msgs)).toEqual([{ role: 'assistant', content: '' }])
  })

  it('handles empty/undefined input defensively', () => {
    expect(buildHistory([])).toEqual([])
    expect(buildHistory(undefined)).toEqual([])
  })
})

describe('assistantContentFor', () => {
  it('returns the reply when non-empty', () => {
    expect(assistantContentFor('Here is your plan.')).toBe('Here is your plan.')
  })

  it('substitutes a clear message when reply is empty (the step-limit bug)', () => {
    const out = assistantContentFor('')
    expect(out).not.toBe('')
    expect(out.toLowerCase()).toContain("couldn't")
  })

  it('treats whitespace-only / nullish reply as empty', () => {
    expect(assistantContentFor('   ')).toBe(assistantContentFor(''))
    expect(assistantContentFor(null)).toBe(assistantContentFor(''))
    expect(assistantContentFor(undefined)).toBe(assistantContentFor(''))
  })

  it('never returns an empty assistant bubble', () => {
    for (const v of ['', '  ', null, undefined, '\n']) {
      expect(assistantContentFor(v).trim().length).toBeGreaterThan(0)
    }
  })
})
