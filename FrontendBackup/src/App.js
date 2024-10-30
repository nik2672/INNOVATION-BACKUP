// App.js
import React, { useState, useMemo } from 'react';
import {
  AppBar, Toolbar, Typography, Container, Button, Box,
  Drawer, List, ListItem, ListItemIcon, ListItemText, IconButton,
  Switch, Snackbar, Alert, Divider,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Home as HomeIcon,
  Info as InfoIcon,
  Mail as MailIcon,
  Brightness4 as Brightness4Icon,
  Brightness7 as Brightness7Icon,
} from '@mui/icons-material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { red } from '@mui/material/colors';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';

// Import your components
import About from './About'; // Assume About component is in About.js
import Contact from './Contact'; // Assume Contact component is in Contact.js
import PredictionForm from './PredictionForm'; // Assume PredictionForm component is in PredictionForm.js
import PredictionPage from './PredictionPage'; // New PredictionPage component

function App() {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const [snackbarOpen, setSnackbarOpen] = useState(false);

  // Create a theme that uses the dark mode setting and sets primary color to red
  const theme = useMemo(() => createTheme({
    palette: {
      mode: darkMode ? 'dark' : 'light',
      primary: {
        main: red[500],
      },
    },
  }), [darkMode]);

  const toggleDrawer = (open) => (event) => {
    if (
      event &&
      event.type === 'keydown' &&
      (event.key === 'Tab' || event.key === 'Shift')
    ) {
      return;
    }
    setDrawerOpen(open);
  };

  const handleDarkModeToggle = () => {
    setDarkMode(!darkMode);
    setSnackbarOpen(true);
  };

  const handleSnackbarClose = (event, reason) => {
    if (reason === 'clickaway') {
      return;
    }
    setSnackbarOpen(false);
  };

  const drawerContent = (
    <Box
      sx={{ width: 250 }}
      role="presentation"
      onClick={toggleDrawer(false)}
      onKeyDown={toggleDrawer(false)}
    >
      <List>
        <ListItem button component={Link} to="/">
          <ListItemIcon sx={{ color: darkMode ? 'white' : 'inherit' }}>
            <HomeIcon />
          </ListItemIcon>
          <ListItemText primary="Home" sx={{ color: darkMode ? 'white' : 'inherit' }} />
        </ListItem>
        <ListItem button component={Link} to="/about">
          <ListItemIcon sx={{ color: darkMode ? 'white' : 'inherit' }}>
            <InfoIcon />
          </ListItemIcon>
          <ListItemText primary="About" sx={{ color: darkMode ? 'white' : 'inherit' }} />
        </ListItem>
        <ListItem button component={Link} to="/contact">
          <ListItemIcon sx={{ color: darkMode ? 'white' : 'inherit' }}>
            <MailIcon />
          </ListItemIcon>
          <ListItemText primary="Contact" sx={{ color: darkMode ? 'white' : 'inherit' }} />
        </ListItem>
      </List>
      <Divider />
      <List>
        <ListItem>
          <ListItemIcon sx={{ color: darkMode ? 'white' : 'inherit' }}>
            {darkMode ? <Brightness7Icon /> : <Brightness4Icon />}
          </ListItemIcon>
          <ListItemText primary="Dark Mode" sx={{ color: darkMode ? 'white' : 'inherit' }} />
          <Switch checked={darkMode} onChange={handleDarkModeToggle} />
        </ListItem>
      </List>
    </Box>
  );

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
          <AppBar position="static" color="primary" enableColorOnDark>
            <Toolbar>
              <IconButton
                edge="start"
                color="inherit"
                aria-label="menu"
                onClick={toggleDrawer(true)}
              >
                <MenuIcon />
              </IconButton>
              <Typography variant="h6" sx={{ flexGrow: 1 }}>
                Real Estate Price Predictor
              </Typography>
              <Button color="inherit" component={Link} to="/about">
                About
              </Button>
              <Button color="inherit" component={Link} to="/contact">
                Contact
              </Button>
            </Toolbar>
          </AppBar>

          <Drawer anchor="left" open={drawerOpen} onClose={toggleDrawer(false)}>
            {drawerContent}
          </Drawer>

          <Routes>
            <Route path="/" element={<PredictionForm />} />
            <Route path="/about" element={<About />} />
            <Route path="/contact" element={<Contact />} />
            <Route path="/prediction" element={<PredictionPage />} />
          </Routes>

          <Box
            component="footer"
            sx={{ bgcolor: 'background.paper', py: 6, mt: 'auto', textAlign: 'center' }}
          >
            <Container maxWidth="lg">
              <Typography variant="body1" align="center">
                Real Estate Price Predictor
              </Typography>
              <Typography variant="body2" color="text.secondary" align="center">
                {'© '}
                {new Date().getFullYear()} High Distinction Only Group 14 - 05
              </Typography>
              <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center' }}>
                <a href="https://www.facebook.com/" target="_blank" rel="noopener noreferrer">
                  <img src="/images/icon1.png" alt="Facebook" style={{ width: 24, margin: '0 8px' }} />
                </a>
                <a href="https://www.instagram.com/" target="_blank" rel="noopener noreferrer">
                  <img src="/images/icon2.png" alt="Instagram" style={{ width: 24, margin: '0 8px' }} />
                </a>
                <a href="https://x.com/" target="_blank" rel="noopener noreferrer">
                  <img src="/images/icon3.png" alt="X" style={{ width: 24, margin: '0 8px' }} />
                </a>
              </Box>
            </Container>
          </Box>

          <Snackbar
            open={snackbarOpen}
            autoHideDuration={6000}
            onClose={handleSnackbarClose}
          >
            <Alert
              onClose={handleSnackbarClose}
              severity="success"
              sx={{ width: '100%' }}
            >
              {darkMode ? 'Dark mode enabled!' : 'Light mode enabled!'}
            </Alert>
          </Snackbar>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;