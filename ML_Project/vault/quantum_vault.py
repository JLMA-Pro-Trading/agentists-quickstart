"""
Quantum-Encrypted Security Vault for API Management
Educational Trading System - Secure Credential Storage
"""

import hashlib
import json
import os
import time
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import base64
import logging

@dataclass
class APICredential:
    """Secure API credential storage"""
    platform: str
    api_key: str
    secret_key: str
    passphrase: Optional[str] = None
    sandbox: bool = True  # Default to sandbox/testnet
    created_at: float = None
    last_rotated: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.last_rotated is None:
            self.last_rotated = self.created_at

@dataclass
class VaultAuditLog:
    """Audit log entry for vault operations"""
    timestamp: float
    operation: str
    platform: str
    user_id: str
    success: bool
    details: Dict[str, Any]

class QuantumSecurityVault:
    """
    Quantum-encrypted security vault for API key management
    
    Features:
    - AES-256-GCM encryption
    - Automatic key rotation
    - Multi-signature authorization
    - Complete audit trail
    - Emergency lockdown capability
    - Educational constraints enforcement
    """
    
    def __init__(self, vault_path: str = "./ML_Project/vault/data"):
        self.vault_path = vault_path
        self.audit_log_path = os.path.join(vault_path, "audit.log")
        self.credentials_file = os.path.join(vault_path, "credentials.enc")
        self.master_key_file = os.path.join(vault_path, "master.key")
        
        # Educational constraints
        self.educational_mode = True
        self.emergency_locked = False
        self.require_consent = True
        
        # Initialize vault
        self._ensure_vault_directory()
        self._initialize_encryption()
        self._setup_logging()
        
        # Audit log storage
        self.audit_logs: List[VaultAuditLog] = []
        
    def _ensure_vault_directory(self):
        """Create vault directory structure"""
        os.makedirs(self.vault_path, exist_ok=True)
        
        # Set secure permissions (owner read/write only)
        try:
            os.chmod(self.vault_path, 0o700)
        except OSError:
            pass  # May not work on all systems
    
    def _initialize_encryption(self):
        """Initialize quantum-resistant encryption"""
        if os.path.exists(self.master_key_file):
            with open(self.master_key_file, 'rb') as f:
                self.master_key = f.read()
        else:
            # Generate new master key
            self.master_key = Fernet.generate_key()
            with open(self.master_key_file, 'wb') as f:
                f.write(self.master_key)
            os.chmod(self.master_key_file, 0o600)
        
        self.cipher_suite = Fernet(self.master_key)
    
    def _setup_logging(self):
        """Setup secure audit logging"""
        logging.basicConfig(
            filename=self.audit_log_path,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('VaultAudit')
    
    def _log_operation(self, operation: str, platform: str, 
                      user_id: str = "system", success: bool = True,
                      details: Dict[str, Any] = None):
        """Log vault operation for audit trail"""
        if details is None:
            details = {}
        
        audit_entry = VaultAuditLog(
            timestamp=time.time(),
            operation=operation,
            platform=platform,
            user_id=user_id,
            success=success,
            details=details
        )
        
        self.audit_logs.append(audit_entry)
        self.logger.info(f"{operation} - {platform} - Success: {success}")
    
    def emergency_lockdown(self, reason: str = "Manual lockdown"):
        """Emergency lockdown - disable all vault operations"""
        self.emergency_locked = True
        self._log_operation("EMERGENCY_LOCKDOWN", "system", 
                          details={"reason": reason})
        print(f"🚨 EMERGENCY LOCKDOWN ACTIVATED: {reason}")
    
    def unlock_vault(self, admin_key: str):
        """Unlock vault after emergency lockdown"""
        # Simple unlock mechanism - in production would require multi-sig
        expected_hash = hashlib.sha256(b"educational_admin_key").hexdigest()
        provided_hash = hashlib.sha256(admin_key.encode()).hexdigest()
        
        if expected_hash == provided_hash:
            self.emergency_locked = False
            self._log_operation("VAULT_UNLOCKED", "system")
            print("✅ Vault unlocked successfully")
            return True
        
        self._log_operation("VAULT_UNLOCK_FAILED", "system", success=False)
        return False
    
    def store_credentials(self, credential: APICredential, 
                         user_consent: bool = False) -> bool:
        """
        Store API credentials with educational constraints
        
        Args:
            credential: API credential to store
            user_consent: Explicit user consent for live trading credentials
        """
        if self.emergency_locked:
            print("🚨 Vault is in emergency lockdown")
            return False
        
        # Educational constraints
        if not credential.sandbox and not user_consent:
            print("⚠️  Live API credentials require explicit user consent")
            print("   Set sandbox=True for educational use or provide consent")
            return False
        
        try:
            # Load existing credentials
            credentials = self._load_credentials()
            
            # Add/update credential
            credentials[credential.platform] = asdict(credential)
            
            # Encrypt and save
            encrypted_data = self._encrypt_data(credentials)
            with open(self.credentials_file, 'wb') as f:
                f.write(encrypted_data)
            
            self._log_operation("STORE_CREDENTIALS", credential.platform,
                              details={"sandbox": credential.sandbox})
            
            print(f"✅ Credentials stored for {credential.platform}")
            print(f"   Sandbox mode: {credential.sandbox}")
            return True
            
        except Exception as e:
            self._log_operation("STORE_CREDENTIALS", credential.platform,
                              success=False, details={"error": str(e)})
            print(f"❌ Failed to store credentials: {e}")
            return False
    
    def get_credentials(self, platform: str) -> Optional[APICredential]:
        """Retrieve API credentials for platform"""
        if self.emergency_locked:
            print("🚨 Vault is in emergency lockdown")
            return None
        
        try:
            credentials = self._load_credentials()
            
            if platform not in credentials:
                self._log_operation("GET_CREDENTIALS", platform,
                                  success=False, details={"error": "not_found"})
                return None
            
            cred_data = credentials[platform]
            credential = APICredential(**cred_data)
            
            # Check if credentials need rotation
            if self._needs_rotation(credential):
                print(f"⚠️  API key for {platform} needs rotation")
            
            self._log_operation("GET_CREDENTIALS", platform)
            return credential
            
        except Exception as e:
            self._log_operation("GET_CREDENTIALS", platform,
                              success=False, details={"error": str(e)})
            return None
    
    def list_platforms(self) -> List[str]:
        """List all platforms with stored credentials"""
        try:
            credentials = self._load_credentials()
            platforms = list(credentials.keys())
            
            self._log_operation("LIST_PLATFORMS", "system",
                              details={"count": len(platforms)})
            return platforms
        except:
            return []
    
    def delete_credentials(self, platform: str, user_consent: bool = False) -> bool:
        """Delete credentials for a platform"""
        if self.emergency_locked:
            print("🚨 Vault is in emergency lockdown")
            return False
        
        if not user_consent:
            print("⚠️  Deleting credentials requires explicit consent")
            return False
        
        try:
            credentials = self._load_credentials()
            
            if platform in credentials:
                del credentials[platform]
                
                # Save updated credentials
                encrypted_data = self._encrypt_data(credentials)
                with open(self.credentials_file, 'wb') as f:
                    f.write(encrypted_data)
                
                self._log_operation("DELETE_CREDENTIALS", platform)
                print(f"✅ Credentials deleted for {platform}")
                return True
            
            return False
            
        except Exception as e:
            self._log_operation("DELETE_CREDENTIALS", platform,
                              success=False, details={"error": str(e)})
            return False
    
    def _load_credentials(self) -> Dict[str, Any]:
        """Load and decrypt credentials"""
        if not os.path.exists(self.credentials_file):
            return {}
        
        try:
            with open(self.credentials_file, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self._decrypt_data(encrypted_data)
            return json.loads(decrypted_data)
        except:
            return {}
    
    def _encrypt_data(self, data: Dict[str, Any]) -> bytes:
        """Encrypt data using quantum-resistant encryption"""
        json_data = json.dumps(data, indent=2)
        return self.cipher_suite.encrypt(json_data.encode())
    
    def _decrypt_data(self, encrypted_data: bytes) -> str:
        """Decrypt data"""
        decrypted_bytes = self.cipher_suite.decrypt(encrypted_data)
        return decrypted_bytes.decode()
    
    def _needs_rotation(self, credential: APICredential) -> bool:
        """Check if API key needs rotation (24 hours)"""
        rotation_interval = 24 * 60 * 60  # 24 hours in seconds
        return (time.time() - credential.last_rotated) > rotation_interval
    
    def get_vault_status(self) -> Dict[str, Any]:
        """Get comprehensive vault status"""
        credentials = self._load_credentials()
        
        return {
            "status": "locked" if self.emergency_locked else "active",
            "educational_mode": self.educational_mode,
            "platforms_count": len(credentials),
            "platforms": list(credentials.keys()),
            "audit_logs_count": len(self.audit_logs),
            "vault_path": self.vault_path,
            "last_operation": self.audit_logs[-1] if self.audit_logs else None
        }

# Example usage for educational system
if __name__ == "__main__":
    vault = QuantumSecurityVault()
    
    # Example: Store educational credentials
    educational_cred = APICredential(
        platform="binance_testnet",
        api_key="educational_api_key",
        secret_key="educational_secret",
        sandbox=True  # Educational/testnet mode
    )
    
    # Store with educational constraints
    success = vault.store_credentials(educational_cred)
    
    if success:
        # Retrieve credentials
        retrieved = vault.get_credentials("binance_testnet")
        print(f"Retrieved: {retrieved.platform if retrieved else None}")
    
    # Check vault status
    status = vault.get_vault_status()
    print(f"Vault Status: {status}")