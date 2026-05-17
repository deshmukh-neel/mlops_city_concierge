import { afterEach, describe, expect, it, vi } from 'vitest'
import { sendMessage } from './chat'

function mockFetchOnce(body, ok = true, status = 200) {
  globalThis.fetch = vi.fn().mockResolvedValue({
    ok,
    status,
    json: async () => body,
  })
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('sendMessage — /chat adapter', () => {
  it('POSTs to /chat with message + history (not /predict)', async () => {
    mockFetchOnce({ reply: 'hi', places: [], ragLabel: 'agent' })
    await sendMessage('best tacos', [{ role: 'user', content: 'hello' }])

    const [url, opts] = globalThis.fetch.mock.calls[0]
    expect(url).toMatch(/\/chat$/)
    expect(opts.method).toBe('POST')
    const sent = JSON.parse(opts.body)
    expect(sent.message).toBe('best tacos')
    expect(sent.history).toEqual([{ role: 'user', content: 'hello' }])
  })

  it('maps PlaceCard shape (with coords) into the UI place shape', async () => {
    mockFetchOnce({
      reply: 'Here is your night.',
      ragLabel: '3 stops',
      places: [
        {
          place_id: 'abc',
          name: 'Trick Dog',
          address: '3010 20th St',
          rating: 4.6,
          primary_type: 'bar',
          latitude: 37.7592,
          longitude: -122.4106,
          rationale: 'Great cocktails',
        },
      ],
    })

    const out = await sendMessage('plan my night')

    expect(out.reply).toContain('Here is your night.')
    expect(out.ragLabel).toBe('3 stops')
    expect(out.places).toHaveLength(1)
    const p = out.places[0]
    expect(p.id).toBe('abc')
    expect(p.name).toBe('Trick Dog')
    expect(p.latitude).toBe(37.7592)
    expect(p.longitude).toBe(-122.4106)
    expect(p.num).toBe('1')
    expect(p.rating).toBe('4.6')
  })

  it('preserves stop order as route order', async () => {
    mockFetchOnce({
      reply: 'r',
      places: [
        { place_id: 'a', name: 'A', rationale: 'x' },
        { place_id: 'b', name: 'B', rationale: 'x' },
        { place_id: 'c', name: 'C', rationale: 'x' },
      ],
    })

    const out = await sendMessage('q')
    expect(out.places.map((p) => p.id)).toEqual(['a', 'b', 'c'])
    expect(out.places.map((p) => p.num)).toEqual(['1', '2', '3'])
  })

  it('keeps a coordinate-less stop with null lat/lng (not dropped)', async () => {
    mockFetchOnce({
      reply: 'r',
      places: [{ place_id: 'ed', name: 'Editorial Spot', rationale: 'x' }],
    })

    const out = await sendMessage('q')
    expect(out.places).toHaveLength(1)
    expect(out.places[0].latitude).toBeNull()
    expect(out.places[0].longitude).toBeNull()
  })

  it('throws a useful error on non-OK response', async () => {
    mockFetchOnce({ detail: 'agent unavailable' }, false, 503)
    await expect(sendMessage('q')).rejects.toThrow(/503.*agent unavailable/)
  })
})
