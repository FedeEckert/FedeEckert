// Footer year
document.getElementById('year').textContent = new Date().getFullYear();

// Mobile menu
function toggleNav(){
  const nav = document.getElementById('navLinks');
  const visible = getComputedStyle(nav).display !== 'none';
  nav.style.display = visible ? 'none' : 'block';
}
