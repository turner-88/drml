package cache

import (
	"sync"
	"time"
)

type entry struct {
	value     string
	expiresAt time.Time
}

// ContentCache is a simple in-memory TTL cache for content table rows.
type ContentCache struct {
	items sync.Map
	ttl   time.Duration
}

// New creates a ContentCache with the given TTL.
func New(ttl time.Duration) *ContentCache {
	return &ContentCache{ttl: ttl}
}

// Get returns the cached value and true if it exists and hasn't expired.
func (c *ContentCache) Get(key string) (string, bool) {
	raw, ok := c.items.Load(key)
	if !ok {
		return "", false
	}
	e := raw.(entry)
	if time.Now().After(e.expiresAt) {
		c.items.Delete(key)
		return "", false
	}
	return e.value, true
}

// Set stores a value with the configured TTL.
func (c *ContentCache) Set(key, value string) {
	c.items.Store(key, entry{
		value:     value,
		expiresAt: time.Now().Add(c.ttl),
	})
}

// Invalidate removes a key from the cache.
func (c *ContentCache) Invalidate(key string) {
	c.items.Delete(key)
}
