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


<script>
(function(){
  const wrap  = document.getElementById('researchCarousel');
  if(!wrap) return;

  const track = wrap.querySelector('.cards-track');
  const prev  = wrap.querySelector('.cards-arrow.prev');
  const next  = wrap.querySelector('.cards-arrow.next');
  const gap   = parseFloat(getComputedStyle(track).gap) || 16;

  function cardStep(){
    const first = track.querySelector('.card');
    if(!first) return 0;
    return first.getBoundingClientRect().width + gap;
  }

  function updateArrows(){
    const max = track.scrollWidth - track.clientWidth - 1;
    prev.disabled = track.scrollLeft <= 0;
    next.disabled = track.scrollLeft >= max;
  }

  function go(dir){
    track.scrollBy({ left: dir * cardStep(), behavior: 'smooth' });
    setTimeout(updateArrows, 350);
  }

  prev.addEventListener('click', () => go(-1));
  next.addEventListener('click', () => go(1));
  track.addEventListener('scroll', updateArrows, { passive: true });
  window.addEventListener('resize', updateArrows);

  updateArrows();
})();
</script>
