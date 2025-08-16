import React, { useState } from "react";
import axios from "axios";
import {
  Box,
  TextField,
  Typography,
  Button,
  Container,
  Card,
  CardContent,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from "@mui/material";
import { styled } from "@mui/system";
import { People } from "@mui/icons-material";

// Gradient background with an image
const GradientBackground = styled(Box)({
  minHeight: "100vh",
  background: "url('https://pbs.twimg.com/media/GZiZZZRbgAA3By-?format=jpg&name=4096x4096') no-repeat center center fixed",
  backgroundSize: "cover",
  display: "flex",
  justifyContent: "center",
  alignItems: "center",
  overflow: "hidden",
});

// Glass-style card for form elements
const GlassCard = styled(Card)({
  backdropFilter: "blur(20px)",
  backgroundColor: "rgba(255, 255, 255, 0.2)",
  border: "1px solid rgba(255, 255, 255, 0.3)",
  boxShadow: "0 4px 30px rgba(0, 0, 0, 0.2)",
  borderRadius: "20px",
  overflow: "hidden",
});

// Button with gradient background and black text
const StyledButton = styled(Button)({
  marginTop: "20px",
  background: "linear-gradient(to right, navy, white)",
  color: "#000",
  fontWeight: "bold",
  padding: "10px 20px",
  borderRadius: "50px",
  "&:hover": {
    background: "linear-gradient(to right, darkgreen, lightgreen)",
  },
  transition: "all 0.3s ease",
});

// Text field with black text and appropriate borders
const StyledTextField = styled(TextField)({
  "& .MuiOutlinedInput-root": {
    "& fieldset": { borderColor: "#000" },
    "&:hover fieldset": { borderColor: "darkgreen" },
    "&.Mui-focused fieldset": { borderColor: "lightgreen" },
  },
  "& .MuiInputBase-input": { color: "#000" },
  "& .MuiInputLabel-root": { color: "#000" },
});

// Typography with black text
const FancyTypography = styled(Typography)({
  fontFamily: "Poppins, sans-serif",
  fontWeight: "600",
  textShadow: "1px 1px 2px rgba(255, 255, 255, 0.5)",
  color: "#000",
  whiteSpace: "nowrap",
  overflow: "hidden",
  textOverflow: "ellipsis",
});

// Popup Dialog Component
const ResultDialog = ({ open, onClose, data }) => (
  <Dialog open={open} onClose={onClose}>
    <DialogTitle>Detection Results</DialogTitle>
    <DialogContent>
      <Typography variant="body1">
        <strong>Bot Probability:</strong> {data.botProbability}
      </Typography>
      <Typography variant="body1">
        <strong>Status:</strong> {data.status}
      </Typography>
      <Typography variant="body1">
        <strong>Message:</strong> {data.message}
      </Typography>
    </DialogContent>
    <DialogActions>
      <Button onClick={onClose} color="primary">
        Close
      </Button>
    </DialogActions>
  </Dialog>
);

const SocialMediaBotDetection = () => {
  const [username, setUsername] = useState("");
  const [resultData, setResultData] = useState({
    botProbability: "",
    status: "",
    message: "",
  });
  const [isDialogOpen, setIsDialogOpen] = useState(false);

  const handleInputChange = (event) => {
    setUsername(event.target.value);
  };

  const handleSubmit = async () => {
    try {
      // Include account_name as a query parameter in the URL
      const response = await axios.post(`http://127.0.0.1:8000/detect-bot/?account_name=${username}`);

      setResultData({
        botProbability: response.data.bot_probability,
        status: response.data.status,
        message: response.data.message,
      });

      setIsDialogOpen(true); // Open dialog with result
    } catch (error) {
      console.error("Error detecting bot:", error);
      alert("An error occurred. Please check your input and try again.");
    }
  };

  const handleCloseDialog = () => {
    setIsDialogOpen(false);
  };

  return (
    <GradientBackground>
      <Container maxWidth="sm">
        <GlassCard>
          <CardContent sx={{ padding: "30px", textAlign: "center" }}>
            <FancyTypography variant="h4" gutterBottom>
              🕵🏻‍♂️ Bot Accounts Detector 🕵🏻‍♂️
            </FancyTypography>
            <Box component="form" noValidate autoComplete="off">
              <StyledTextField
                label="Account Username"
                fullWidth
                margin="normal"
                variant="outlined"
                value={username}
                onChange={handleInputChange}
                InputProps={{
                  startAdornment: (
                    <People sx={{ color: "#000" }} />
                  ),
                }}
              />
              <StyledButton variant="contained" fullWidth onClick={handleSubmit}>
                🚀 Detect Bot
              </StyledButton>
            </Box>
          </CardContent>
        </GlassCard>
      </Container>
      <ResultDialog open={isDialogOpen} onClose={handleCloseDialog} data={resultData} />
    </GradientBackground>
  );
};

export default SocialMediaBotDetection;
