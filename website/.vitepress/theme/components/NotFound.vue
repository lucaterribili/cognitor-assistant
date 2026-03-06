<template>
  <div class="not-found">
    <div class="not-found__content">
      <p class="not-found__code">404</p>
      <h1 class="not-found__title">Pagina non trovata</h1>
      <p class="not-found__description">
        La pagina che stai cercando non esiste o è stata spostata.
      </p>
      <div class="not-found__actions">
        <a href="/" class="not-found__btn not-found__btn--primary">
          Torna alla Home
        </a>
        <a
          href="https://github.com/lucaterribili/cognitor-assistant/issues"
          class="not-found__btn not-found__btn--secondary"
          target="_blank"
          rel="noopener noreferrer"
        >
          Segnala un problema
        </a>
      </div>
    </div>
  </div>
</template>

<style scoped>
.not-found {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: calc(100vh - var(--vp-nav-height));
  padding: 2rem;
  text-align: center;
}

.not-found__content {
  max-width: 480px;
}

.not-found__code {
  font-size: 6rem;
  font-weight: 700;
  line-height: 1;
  color: var(--vp-c-brand-1);
  margin-bottom: 1rem;
}

.not-found__title {
  font-size: 1.75rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
  margin-bottom: 1rem;
}

.not-found__description {
  color: var(--vp-c-text-2);
  margin-bottom: 2rem;
  line-height: 1.6;
}

.not-found__actions {
  display: flex;
  gap: 1rem;
  justify-content: center;
  flex-wrap: wrap;
}

.not-found__btn {
  display: inline-block;
  padding: 0.625rem 1.25rem;
  border-radius: 8px;
  font-size: 0.875rem;
  font-weight: 500;
  text-decoration: none;
  transition: background-color 0.2s, color 0.2s;
}

.not-found__btn--primary {
  background-color: var(--vp-c-brand-1);
  color: #fff;
}

.not-found__btn--primary:hover {
  background-color: var(--vp-c-brand-2);
}

.not-found__btn--secondary {
  background-color: var(--vp-c-bg-soft);
  color: var(--vp-c-text-1);
  border: 1px solid var(--vp-c-divider);
}

.not-found__btn--secondary:hover {
  background-color: var(--vp-c-bg-mute);
}
</style>
