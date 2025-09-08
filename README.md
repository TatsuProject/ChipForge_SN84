# ChipForge - Subnet # 84

ChipForge is a Bittensor subnet that incentivizes the development of optimized hardware designs through competitive evaluation and scoring. The subnet challenges participants to create efficient FPGA and ASIC implementations of various digital circuits, rewarding those who achieve the best performance across multiple metrics.

## Overview

ChipForge operates as a competitive platform where:
- **Miners** submit hardware design solutions (Verilog/SystemVerilog)
- **Validators** evaluate submissions using industry-standard EDA tools
- **Challenges** are rotated periodically with different design requirements
- **Rewards** are distributed based on performance metrics and competitive scoring

## Architecture

### Core Components

1. **Challenge Server** - Manages challenges, submissions, and evaluations
2. **Miners** - Submit optimized hardware designs for active challenges
3. **Validators** - Download, evaluate, and score miner submissions
4. **EDA Server** - Performs synthesis, place & route, and timing analysis

### Workflow

```
1. Challenge Activation → 2. Miner Submissions → 3. Batch Creation → 
4. Validator Downloads → 5. EDA Evaluation → 6. Score Submission → 
7. Weight Setting → 8. Challenge Completion
```

## Getting Started

### Prerequisites

- Python 3.12
- Bittensor wallet with registered hotkey
- Access to challenge server API
- EDA tools (for validators)

### For Miners

#### Installation

```bash
# Clone the repository
git clone <repository-url>
cd chipforge-subnet

# Install dependencies
pip install -r requirements.txt

# Configure your wallet
btcli wallet create --wallet.name miner --wallet.hotkey default
```

#### Submitting Solutions

```bash
# set wallet and path to zip file in .env file and run this
./submit_score.sh
```

#### Solution Format

Solutions must be packaged as ZIP files containing:
- Verilog/SystemVerilog source files (.v, .sv)
- Testbench files (optional)
- Constraint files (optional)
- README with design description (optional)

#### Rate Limits
- Challenge server accepts 5 requests per IP per hour.
- One hotkey is allowed to submit a maximum 5 solutions for a specific challenge.

### For Validators

#### Installation

```bash
# Install validator dependencies
pip install -r validator_requirements.txt

# Configure EDA tools (Vivado, Quartus, etc.)
# Set up validator secrets and API keys
```

#### Running a Validator

```bash
# Set parameters in .env file and run
./start_validator
```

#### Validator Responsibilities

- Download submissions from active batches
- Evaluate designs using EDA tools
- Submit scores based on multiple metrics:
  - **Functionality** (0-100): Correctness and testbench passing
  - **Area** (0-100): Resource utilization efficiency
  - **Delay** (0-100): Timing performance
  - **Power** (0-100): Power consumption optimization
  - **Overall** (0-100): Weighted combination of all metrics

## Evaluation Metrics

### Scoring System

Each submission is evaluated across four key metrics:

1. **Functionality Score** (100% weight)
   - Testbench pass/fail status
   - Functional correctness verification
   - Compliance with specifications

2. **Area Score** (TBD)
   - LUT utilization
   - Register usage
   - Memory block efficiency
   - Overall resource optimization

3. **Delay Score** (TBD)
   - Maximum frequency achieved
   - Critical path timing
   - Setup/hold time margins

4. **Power Score** (TBD)
   - Static power consumption
   - Dynamic power analysis
   - Power efficiency metrics

### Competitive Ranking

- Submissions are ranked by overall score
- Only submissions that beat the current challenge-wide best score receive rewards
- Weights are set to reward the highest-scoring submission
- Emission burning occurs when no submissions exceed quality thresholds

## Challenge Types

### Current Challenge Categories

1. **Digital Signal Processing**
   - FIR/IIR filter implementations
   - FFT/IFFT processors
   - Digital modulators/demodulators

2. **Arithmetic Units**
   - Multipliers and dividers
   - Floating-point units
   - Specialized arithmetic (e.g., modular arithmetic)

3. **Communication Interfaces**
   - UART, SPI, I2C controllers
   - Ethernet MAC implementations
   - Protocol processors

4. **Custom Logic**
   - Application-specific designs
   - Algorithm implementations
   - System-on-chip components

## API Reference

### Miner Endpoints

```
GET  /api/v1/challenges/active              # Get active challenge
POST /api/v1/challenges/{id}/generate-submission-id  # Generate submission ID
POST /api/v1/challenges/{id}/submit         # Submit design solution
GET  /api/v1/challenges/{id}/submissions/hotkey/{hotkey}  # Check submissions
```

### Validator Endpoints

```
GET  /api/v1/challenges/{id}/batch/current  # Get current evaluation batch
GET  /api/v1/challenges/{id}/submissions/{submission_id}/download  # Download submission
POST /api/v1/challenges/{id}/submissions/{submission_id}/submit_score  # Submit evaluation
```

## Configuration

### Batch Management

The subnet uses a dynamic batch system:
- Submissions are grouped into evaluation batches
- Each batch has a download window (10 minutes) and evaluation window (20 minutes)
- Only one batch is exposed to validators at a time
- Batches transition: EXPOSED → EVALUATING → COMPLETED

## Security

### Authentication

- **Signature-based auth**: All API calls require Ed25519 signatures
- **Validator secrets**: Additional secret keys for validator endpoints
- **Hotkey verification**: Ensures submissions come from registered miners
- **Timestamp validation**: Prevents replay attacks

### Data Integrity

- **File hashing**: All submissions are verified with SHA256 hashes
- **Signature verification**: Using Bittensor's native signing methods
- **Audit logging**: Complete audit trail of all operations

## Monitoring

### Health Checks

```bash
# Check challenge server health
curl http://challenge-server:8000/health
```

### Logging

- Structured JSON logging for all components
- Separate log levels for development and production
- Audit logs for security-sensitive operations
- Performance metrics and timing data

## Troubleshooting

### Common Issues

1. **Signature Verification Failed**
   - Ensure using Bittensor's native signing methods
   - Check timestamp accuracy (must be within 10 minutes)
   - Verify hotkey registration on subnet

2. **Submission Upload Failed**
   - Check ZIP file format and contents
   - Verify file size limits (typically 10MB)
   - Ensure proper authentication headers

3. **Validator Download Issues**
   - Confirm validator secret configuration
   - Check batch timing and availability
   - Verify network connectivity to challenge server

4. **EDA Tool Integration**
   - Ensure proper tool licensing and setup
   - Check environment variable configuration
   - Verify design constraints and timing requirements

## Contributing

### Submission Guidelines

1. Follow PEP 8 coding standards
2. Include comprehensive tests
3. Update documentation for new features
4. Ensure backward compatibility

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Discord**: [ChipForge - SN84](https://discord.com/channels/799672011265015819/1408463235082092564)

## Roadmap

### Upcoming Features

- ...

### Version History

- **v1.0.0** - Initial release with challenge download, solution design, solution verification, submitting scores, and giving reward to winner.
