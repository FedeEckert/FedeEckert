// Footer year
document.getElementById('year').textContent = new Date().getFullYear();

// Mobile menu
function toggleNav(){
  const nav = document.getElementById('navLinks');
  const visible = getComputedStyle(nav).display !== 'none';
  nav.style.display = visible ? 'none' : 'block';
}

// Carrusel con flechas para Research
(function(){
  const track = document.getElementById('cardsTrack');
  if(!track) return;

  const prev = document.querySelector('.cards-arrow.prev');
  const next = document.querySelector('.cards-arrow.next');

  const getStep = () => {
    // Un paso = ancho de una card + gap
    const card = track.querySelector('.card');
    if(!card) return 0;
    const styles = getComputedStyle(track);
    const gap = parseFloat(styles.getPropertyValue('--gap')) || 16;
    return card.getBoundingClientRect().width + gap;
  };

  const updateArrows = () => {
    const maxScroll = track.scrollWidth - track.clientWidth - 1; // tolerancia
    prev.disabled = track.scrollLeft <= 0;
    next.disabled = track.scrollLeft >= maxScroll;
  };

  prev.addEventListener('click', () => {
    track.scrollBy({ left: -getStep(), behavior: 'smooth' });
  });
  next.addEventListener('click', () => {
    track.scrollBy({ left:  getStep(), behavior: 'smooth' });
  });

  track.addEventListener('scroll', updateArrows);
  window.addEventListener('resize', updateArrows);

  // Ocultar flechas si no hay overflow (p.ej., solo 3 cards en desktop)
  const toggleVisibility = () => {
    const overflow = track.scrollWidth > track.clientWidth + 1;
    prev.style.display = next.style.display = overflow ? 'grid' : 'none';
    updateArrows();
  };

  // Inicialización
  toggleVisibility();
  // Por si las fuentes reflowean el layout:
  setTimeout(toggleVisibility, 250);
})();
