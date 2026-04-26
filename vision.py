"""
vision.py — Adaptive Vision Engine for Winamax Expresso (HUD)

Stratégie d'ancrage adaptatif :
  1. Détection des cartes communes (rectangles blancs) → ancre principale
  2. Calcul d'un facteur d'échelle (table_scale) depuis ces cartes
  3. Toutes les autres zones déduites en coordonnées relatives à cet ancre
  4. Fonctionne quelle que soit la taille de la fenêtre Winamax

Carte de référence (1936×1056, fenêtre plein écran) :
  board card :  103×154 px   @ y=376
  hero cards  :  71×80  px   @ x=848–991, y=706–786
  pot         :               @ y=577, x=876–990
  blinds      :               @ y=53, x=1653
  action btn  :               @ y=997

Auteur : Gachiakuta / SignaraFast — 2026
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)

# ── Constantes de calibration (fenêtre 1936×1056) ─────────────────────────
REF_CARD_H   = 154      # hauteur de référence d'une carte commune (px)
REF_CARD_W   = 103      # largeur de référence (px)
REF_CARD_GAP = 8        # espace entre cartes (px)

# Offsets relatifs à la carte commune (index 0) — en fractions de card_h
OFF_POT_Y    = 1.28     # pot Y  (vers le bas depuis le bas des cartes communes)
OFF_HERO_Y   = 2.30     # hero box Y (depuis haut des cartes communes)
OFF_OPP1_Y   = -0.90   # opponent 1 Y (au-dessus des cartes communes)

# Noms de rang Winamax → notation standard
_RANK_ALIASES: Dict[str, str] = {
    "1": "A", "11": "J", "12": "Q", "13": "K", "14": "A",
    "0": "T",            # "10" parfois lu "0" ou "10"
    "T": "T", "10": "T",
}

_VALID_RANKS = set("AKQJT98765432")


# ────────────────────────────────────────────────────────────────────────────
@dataclass
class CardInfo:
    rank: str = ""
    suit: str = ""     # "s" | "h" | "d" | "c"  (spade/heart/diamond/club)
    color: str = ""    # "red" | "black"

    def __str__(self):
        return f"{self.rank}{self.suit}" if self.rank else "??"

    @property
    def valid(self):
        return self.rank in _VALID_RANKS


@dataclass
class PokerVisionState:
    """Résultat complet de la vision — prêt pour route_request()."""
    # Cartes
    board_cards:  List[CardInfo] = field(default_factory=list)
    hero_cards:   List[CardInfo] = field(default_factory=list)

    # Stacks / pot
    hero_stack:   Optional[float] = None
    opp_stacks:   List[float]     = field(default_factory=list)
    pot:          Optional[float] = None
    to_call:      Optional[float] = None

    # Contexte
    street:       str = "preflop"   # preflop | flop | turn | river
    position:     str = "unknown"   # BTN | SB | BB | CO | MP | UTG
    players:      int = 2
    blinds:       List[float]       = field(default_factory=lambda: [5.0, 10.0])
    level:        int = 1

    # Actions dispos
    actions:      List[str] = field(default_factory=list)  # fold/check/call/raise
    hero_action:  str = ""   # action affichée (CHECK, FOLD, …)

    # Méta
    hand_desc:    str = ""   # "2 Paires : 9-4", "Hauteur : A", …
    prize_pool:   Optional[float] = None

    # Debug
    _scale:       float = 1.0
    _board_origin: Tuple[int, int] = (0, 0)

    def to_game_state(self) -> Dict[str, Any]:
        """Convertit vers le format attendu par route_request()."""
        hole_cards  = [str(c) for c in self.hero_cards  if c.valid]
        board_cards = [str(c) for c in self.board_cards if c.valid]

        return {
            "hole_cards":    hole_cards,
            "board_cards":   board_cards,
            "pot":           self.pot or 0.0,
            "stack":         self.hero_stack or 0.0,
            "to_call":       self.to_call or 0.0,
            "blinds":        self.blinds,
            "position":      self.position,
            "street":        self.street,
            "num_players":   self.players,
            "action_history": [],
        }


# ────────────────────────────────────────────────────────────────────────────
class ExpressoVision:
    """
    Moteur de vision adaptatif pour Winamax Expresso.

    Usage :
        ev    = ExpressoVision()
        state = ev.extract(image_path)   # ou extract_from_array(np_array)
    """

    def __init__(self, debug: bool = False):
        self.debug = debug
        self._tess_cfg_num = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789,."
        self._tess_cfg_rank = "--psm 10 --oem 3 -c tessedit_char_whitelist=AKQJT234567890"
        self._tess_cfg_free = "--psm 11 --oem 3 -l fra+eng"

    # ── Public API ──────────────────────────────────────────────────────────

    def extract(self, image_path: str) -> PokerVisionState:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        return self.extract_from_array(img)

    def extract_from_array(self, img: np.ndarray) -> PokerVisionState:
        state = PokerVisionState()
        h, w = img.shape[:2]

        # ── 1. Localise la fenêtre Winamax ──────────────────────────────
        win_x, win_y, win_w, win_h = self._find_window(img)
        logger.debug("Window: x=%d y=%d w=%d h=%d", win_x, win_y, win_w, win_h)

        # ── 2. Détecte les cartes communes (ancre principale) ────────────
        board_rects = self._find_board_cards(img, win_x, win_y, win_w, win_h)

        if not board_rects:
            state.street = "preflop"
            # En preflop : positions fixes relatives à la fenêtre
            self._extract_meta(img, state, win_x, win_y, win_w, win_h)
            self._extract_hero_cards(img, state, win_x, win_y, win_w, win_h, scale=1.0,
                                     board_x=win_x + win_w//2, board_y=win_y + win_h//2)
            return state

        # ── 3. Calcule le facteur d'échelle ─────────────────────────────
        card_h   = board_rects[0][3]
        card_w   = board_rects[0][2]
        scale    = card_h / REF_CARD_H
        board_x0 = board_rects[0][0]
        board_y0 = board_rects[0][1]
        state._scale        = scale
        state._board_origin = (board_x0, board_y0)

        # ── 4. Street (nombre de cartes communes) ───────────────────────
        n = len(board_rects)
        state.street = {1: "flop", 2: "flop", 3: "flop", 4: "turn", 5: "river"}.get(n, "flop")

        # ── 5. OCR cartes communes ───────────────────────────────────────
        state.board_cards = [self._ocr_card(img, r, card_w, card_h) for r in board_rects]

        # ── 6. Nombre de joueurs ─────────────────────────────────────────
        state.players = self._detect_num_players(img, win_x, win_y, win_w, win_h, scale)

        # ── 7. Cartes hero ───────────────────────────────────────────────
        self._extract_hero_cards(img, state, win_x, win_y, win_w, win_h, scale,
                                 board_x=board_x0, board_y=board_y0)

        # ── 8. Pot, stacks, blindes, actions ────────────────────────────
        self._extract_meta(img, state, win_x, win_y, win_w, win_h, scale=scale,
                           board_y0=board_y0, card_h=card_h)

        logger.debug("State: board=%s hero=%s pot=%s stack=%s",
                     state.board_cards, state.hero_cards, state.pot, state.hero_stack)
        return state

    # ── Private helpers ─────────────────────────────────────────────────────

    def _find_window(self, img: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Localise la fenêtre Winamax sans OCR.
        Stratégie (par ordre de priorité) :
          1. Détection par couleur de la barre de titre Winamax (bleu foncé ~#1a2a4a)
          2. Fallback par détection du fond de table (tapis vert/feutré caractéristique)
          3. Fallback final → image entière (0, 0, w, h)
        
        Retourne (x, y, w, h) de la zone utile.
        Robuste aux changements de résolution (1280×720 à 2560×1440) et thèmes de table.
        """
        h, w = img.shape[:2]
        logger.debug("Image dimensions: %dx%d", w, h)

        # ── STRATÉGIE 1 : Détection par couleur de la barre de titre ────────────────────
        # La barre de titre Winamax est bleu foncé (~#1a2a4a en hex, soit BGR: ~#4a2a1a)
        # En HSV : Hue ~110-130°, Saturation ~80-150, Value ~40-90
        logger.debug("Stratégie 1: Détection par couleur de la barre de titre")
        
        # Convertir en HSV pour une meilleure segmentation couleur
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Définir les bornes HSV pour le bleu foncé Winamax
        # Ajustement: le bleu foncé a un hue autour de 115-125, saturation moyenne, valeur basse
        lower_blue = np.array([105, 60, 30], dtype=np.uint8)
        upper_blue = np.array([130, 180, 100], dtype=np.uint8)
        
        mask_title = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Chercher une bande horizontale en haut de l'image (dans les 10% supérieurs)
        top_band_height = max(int(h * 0.05), 20)  # minimum 20px
        top_band_height = min(top_band_height, int(h * 0.15))  # maximum 15%
        top_mask = mask_title[:top_band_height, :]
        
        # Projection horizontale pour trouver la bande continue
        horizontal_proj = cv2.reduce(top_mask, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S).flatten()
        
        # Trouver les régions où la projection est significative
        title_y_start = None
        title_y_end = None
        for y in range(top_mask.shape[0]):
            if int(horizontal_proj[y]) > w * 20:  # Au moins 20 pixels bleus sur la ligne
                if title_y_start is None:
                    title_y_start = y
                title_y_end = y
        
        if title_y_start is not None and title_y_end is not None:
            # Vérifier que la hauteur est cohérente avec une barre de titre (20-40px typiquement)
            title_height = title_y_end - title_y_start + 1
            if 15 <= title_height <= 60:  # Tolérance large pour différentes résolutions
                # Trouver les bords gauche/droit par projection verticale sur la bande détectée
                title_band = top_mask[title_y_start:title_y_end+1, :]
                vertical_proj = cv2.reduce(title_band, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S).flatten()
                
                # Trouver le début et la fin de la barre (où il y a suffisamment de pixels)
                title_x_start = None
                title_x_end = None
                threshold = title_height * 5  # Au moins 5 lignes de pixels bleus
                
                for x in range(vertical_proj.shape[0]):
                    if int(vertical_proj[x]) >= threshold:
                        if title_x_start is None:
                            title_x_start = x
                        title_x_end = x
                
                if title_x_start is not None and title_x_end is not None:
                    # Raffiner les bords avec un peu de marge
                    title_x_start = max(0, title_x_start - 5)
                    title_x_end = min(w, title_x_end + 5)
                    
                    window_w = title_x_end - title_x_start
                    window_h = h - title_y_start
                    
                    logger.debug(
                        "Barre de titre détectée: x=%d y=%d w=%d h=%d (hauteur barre=%d)",
                        title_x_start, title_y_start, window_w, window_h, title_height
                    )
                    return (title_x_start, title_y_start, window_w, window_h)
        
        logger.debug("Stratégie 1 échouée: barre de titre non détectée par couleur")

        # ── STRATÉGIE 2 : Détection du fond de table (tapis vert/feutré) ───────────────
        # Le tapis Winamax Expresso a une teinte verte caractéristique
        # En HSV : Hue ~50-70° (vert), Saturation ~40-120, Value ~60-180
        logger.debug("Stratégie 2: Détection par couleur du tapis de table")
        
        # Bornes HSV pour le vert feutré du tapis
        lower_green = np.array([45, 30, 50], dtype=np.uint8)
        upper_green = np.array([75, 150, 200], dtype=np.uint8)
        
        mask_table = cv2.inRange(hsv, lower_green, upper_green)
        
        # Appliquer un nettoyage morphologique pour connecter les régions
        kernel = np.ones((5, 5), np.uint8)
        mask_table = cv2.morphologyEx(mask_table, cv2.MORPH_CLOSE, kernel)
        mask_table = cv2.morphologyEx(mask_table, cv2.MORPH_OPEN, kernel)
        
        # Trouver les contours pour identifier la plus grande zone rectangulaire
        contours, _ = cv2.findContours(mask_table, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Trouver le plus grand contour par aire
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Seuillage: doit représenter au moins 15% de l'image totale
            min_area = w * h * 0.15
            if area >= min_area:
                # Approximer le contour par un rectangle englobant
                x, y, table_w, table_h = cv2.boundingRect(largest_contour)
                
                # Vérifier que les dimensions sont cohérentes avec une table de poker
                # La table devrait occuper une portion significative de l'image
                aspect_ratio = table_w / table_h if table_h > 0 else 0
                
                # Aspect ratio typique d'une table de poker: 1.2 à 2.5
                if 1.0 <= aspect_ratio <= 3.0:
                    logger.debug(
                        "Tapis de table détecté: x=%d y=%d w=%d h=%d (aire=%.0f, aspect=%.2f)",
                        x, y, table_w, table_h, area, aspect_ratio
                    )
                    return (x, y, table_w, table_h)
        
        logger.debug("Stratégie 2 échouée: tapis de table non détecté")

        # ── STRATÉGIE 3 : Fallback → image entière ─────────────────────────────────────
        logger.debug("Stratégie 3 (fallback): Retour de l'image entière")
        return (0, 0, w, h)

    def _find_board_cards(
        self, img: np.ndarray,
        win_x: int, win_y: int, win_w: int, win_h: int
    ) -> List[Tuple[int, int, int, int]]:
        """
        Détecte les cartes communes (rectangles blancs) dans la zone de table.
        Retourne liste de (x, y, w, h) triée par x croissant.
        """
        # Zone d'intérêt : 15%–80% en hauteur, 10%–90% en largeur
        roi_x1 = win_x + int(win_w * 0.10)
        roi_x2 = win_x + int(win_w * 0.90)
        roi_y1 = win_y + int(win_h * 0.15)
        roi_y2 = win_y + int(win_h * 0.75)

        roi = img[roi_y1:roi_y2, roi_x1:roi_x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Seuillage sur les pixels très clairs (cartes blanches)
        _, mask = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
        kernel  = np.ones((4, 4), np.uint8)
        mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cards = []
        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            area    = bw * bh
            aspect  = bw / bh if bh > 0 else 0
            solidity = cv2.contourArea(cnt) / area if area > 0 else 0

            # Filtre : aspect carte ~0.55–0.80, solide, taille cohérente
            if (0.45 < aspect < 0.88
                    and 1500 < area < 40000
                    and solidity > 0.72):
                # Reconvertit en coordonnées image complète
                cards.append((bx + roi_x1, by + roi_y1, bw, bh))

        # Filtre les doublons proches (overlap > 60%)
        cards = self._deduplicate_rects(cards, iou_thr=0.30)

        # Trie par x, limite à 5 cartes
        cards.sort(key=lambda c: c[0])
        return cards[:5]

    def _deduplicate_rects(self, rects, iou_thr=0.3):
        """Supprime les rectangles très proches/overlapping."""
        if not rects:
            return rects
        # Trie par aire décroissante
        rects = sorted(rects, key=lambda r: r[2]*r[3], reverse=True)
        kept = []
        for r in rects:
            if not any(self._iou(r, k) > iou_thr for k in kept):
                kept.append(r)
        return kept

    @staticmethod
    def _iou(a, b):
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1+aw, ay1+ah
        bx2, by2 = bx1+bw, by1+bh
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        union = aw*ah + bw*bh - inter
        return inter / union if union > 0 else 0

    def _ocr_card(self, img, rect, card_w, card_h) -> CardInfo:
        """OCR d'une carte (commune ou trou) → CardInfo."""
        x, y, w, h = rect
        card_img = img[y:y+h, x:x+w]

        # ── Rang : coin supérieur gauche ────────────────────────────────
        rank_zone = card_img[:int(h * 0.38), :int(w * 0.60)]
        rank = self._ocr_rank(rank_zone)

        # ── Couleur : analyse chromatique de la moitié inférieure ──────
        suit_zone = card_img[int(h * 0.42):int(h * 0.78), :int(w * 0.55)]
        color, suit = self._detect_suit(card_img, suit_zone)

        return CardInfo(rank=rank, suit=suit, color=color)

    def _ocr_rank(self, zone: np.ndarray) -> str:
        """OCR du rang depuis un crop de la zone supérieure gauche d'une carte."""
        if zone.size == 0:
            return ""
        # Agrandit 5× pour meilleure précision
        big = cv2.resize(zone, (zone.shape[1] * 5, zone.shape[0] * 5),
                         interpolation=cv2.INTER_LANCZOS4)
        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        # Seuil inversé : texte noir sur fond blanc → texte blanc sur fond noir
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        pil = Image.fromarray(thresh)
        raw = pytesseract.image_to_string(pil, config=self._tess_cfg_rank).strip()

        return self._normalize_rank(raw)

    def _normalize_rank(self, raw: str) -> str:
        """Normalise le texte OCR en rang poker valide."""
        raw = raw.strip().upper()
        if not raw:
            return ""

        # Alias directs
        if raw in _RANK_ALIASES:
            return _RANK_ALIASES[raw]

        # "10" → "T"
        raw_clean = re.sub(r"[^A-Z0-9]", "", raw)
        if "10" in raw_clean:
            return "T"

        # Cherche le premier caractère valide
        for c in raw_clean:
            mapped = _RANK_ALIASES.get(c, c)
            if mapped in _VALID_RANKS:
                return mapped

        return ""

    def _detect_suit(
        self, card_img: np.ndarray, suit_zone: np.ndarray
    ) -> Tuple[str, str]:
        """
        Détecte couleur + enseigne.
        Retourne ("red"|"black", "h"|"d"|"s"|"c")
        """
        if suit_zone.size == 0:
            return ("black", "")

        # Couleur rouge ou noire
        r = float(suit_zone[:, :, 2].mean())  # Red (BGR)
        g = float(suit_zone[:, :, 1].mean())
        b = float(suit_zone[:, :, 0].mean())
        is_red = (r > g + 15) and (r > b + 15)
        color = "red" if is_red else "black"

        # Enseigne : analyse de forme (croix, cœur, losange, pique)
        suit = self._detect_suit_shape(card_img, is_red)
        return (color, suit)

    def _detect_suit_shape(self, card_img: np.ndarray, is_red: bool) -> str:
        """
        Détecte l'enseigne par analyse des contours dans la zone du symbole.
        Heuristique basée sur ratio hauteur/largeur et compacité.
        """
        h, w = card_img.shape[:2]
        # Cible : symbole d'enseigne dans la zone 30%–70% en hauteur, 5%–55% en largeur
        sym = card_img[int(h*0.32):int(h*0.68), int(w*0.02):int(w*0.52)]
        if sym.size == 0:
            return "h" if is_red else "s"

        gray = cv2.cvtColor(sym, cv2.COLOR_BGR2GRAY)
        # Isole le symbole foncé sur fond blanc
        _, mask = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not cnts:
            return "h" if is_red else "s"

        # Prend le plus grand contour
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < 20:
            return "h" if is_red else "s"

        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity  = area / hull_area if hull_area > 0 else 0
        bx, by, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / bh if bh > 0 else 1.0

        if is_red:
            # Cœur (♥) : solide ~0.72–0.85, aspect ~0.95–1.15, forme concave en haut
            # Carreau (♦) : solide ~0.80–0.95, aspect ~0.70–0.85, forme losange
            if aspect < 0.87 and solidity > 0.78:
                return "d"   # ♦ losange
            else:
                return "h"   # ♥ cœur
        else:
            # Pique (♠) : solide ~0.55–0.72, aspect ~0.85–1.05
            # Trèfle (♣) : solide ~0.55–0.72, aspect ~0.80–1.00, 3 lobes
            if solidity < 0.65:
                return "c"   # ♣ trèfle (moins compact)
            else:
                return "s"   # ♠ pique

    def _extract_hero_cards(
        self, img: np.ndarray, state: PokerVisionState,
        win_x: int, win_y: int, win_w: int, win_h: int,
        scale: float, board_x: int, board_y: int
    ) -> None:
        """
        Localise et OCR les cartes du hero.
        En Winamax Expresso, les cartes hero sont affichées dans une box
        au bas de la table, côte à côte dans un widget ~143px × 80px (ref 1936×1056).
        """
        # Position adaptative : centrée horizontalement, sous les cartes communes
        ref_hero_x = 848       # en coordonnées de référence
        ref_hero_y = 706       # en coordonnées de référence
        ref_hero_w = 143       # largeur totale (2 cartes)
        ref_hero_h = 80        # hauteur

        # Calcul depuis la fenêtre et l'échelle
        # Ratio par rapport à la largeur de référence (1936px)
        ref_img_w  = 1936
        x_ratio    = board_x / ref_img_w

        hero_cx = win_x + int(win_w * x_ratio)   # Centre horizontal approx
        hero_w  = int(ref_hero_w * scale)
        hero_h  = int(ref_hero_h * scale)
        hero_x  = hero_cx - hero_w // 2
        hero_y  = board_y + int(REF_CARD_H * scale) + int(110 * scale)

        # Clip aux limites image
        img_h, img_w = img.shape[:2]
        hero_x = max(0, min(hero_x, img_w - hero_w))
        hero_y = max(0, min(hero_y, img_h - hero_h))

        if self.debug:
            logger.debug("Hero cards zone: x=%d y=%d w=%d h=%d", hero_x, hero_y, hero_w, hero_h)

        hero_area = img[hero_y:hero_y + hero_h, hero_x:hero_x + hero_w]
        if hero_area.size == 0:
            return

        # OCR de tout le bloc (donne "109" pour 10-9, "AA" pour A-A, etc.)
        raw_text = self._ocr_hero_block(hero_area)
        parsed   = self._parse_hero_block(raw_text)

        # Si le bloc OCR échoue, essaie sur chaque demi-bloc
        if len(parsed) < 2:
            half = hero_w // 2
            for i in range(2):
                sub = hero_area[:, i*half:(i+1)*half]
                sub_gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(sub_gray, 100, 255, cv2.THRESH_BINARY_INV)
                sub_big   = cv2.resize(thresh, (thresh.shape[1]*5, thresh.shape[0]*5),
                                       interpolation=cv2.INTER_LANCZOS4)
                raw_sub   = pytesseract.image_to_string(
                    Image.fromarray(sub_big), config=self._tess_cfg_rank
                ).strip()
                rank = self._normalize_rank(raw_sub)
                if rank and (not parsed or parsed[-1].rank != rank):
                    # Couleur du demi-bloc
                    r = sub[:, :, 2].mean()
                    b = sub[:, :, 0].mean()
                    color = "red" if r > b + 20 else "black"
                    suit  = "h" if color == "red" else "s"
                    parsed.append(CardInfo(rank=rank, color=color, suit=suit))

        state.hero_cards = parsed[:2]

    def _ocr_hero_block(self, hero_area: np.ndarray) -> str:
        """OCR du bloc hero (2 cartes côte à côte)."""
        big  = cv2.resize(hero_area, (hero_area.shape[1]*4, hero_area.shape[0]*4),
                          interpolation=cv2.INTER_LANCZOS4)
        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        return pytesseract.image_to_string(
            Image.fromarray(thresh), config=self._tess_cfg_rank
        ).strip()

    def _parse_hero_block(self, raw: str) -> List[CardInfo]:
        """
        Parse "109" → [10, 9], "AA" → [A, A], "AK" → [A, K], etc.
        """
        raw = re.sub(r"[^A-Z0-9]", "", raw.upper())
        if not raw:
            return []

        ranks = []
        i = 0
        while i < len(raw):
            # Teste "10" en premier (deux caractères)
            if i + 1 < len(raw) and raw[i:i+2] == "10":
                ranks.append("T")
                i += 2
            elif raw[i] in "AKQJT":
                ranks.append(raw[i])
                i += 1
            elif raw[i] in "23456789":
                ranks.append(raw[i])
                i += 1
            else:
                i += 1

        return [CardInfo(rank=r, suit="", color="") for r in ranks[:2]]

    def _detect_num_players(
        self, img: np.ndarray,
        win_x: int, win_y: int, win_w: int, win_h: int,
        scale: float
    ) -> int:
        """
        Détecte 2 ou 3 joueurs en cherchant une 2ème box adversaire
        (coin supérieur droit de la fenêtre).
        """
        # Zone opponent droit : x 65–85%, y 12–30%
        opp2_x1 = win_x + int(win_w * 0.65)
        opp2_x2 = win_x + int(win_w * 0.88)
        opp2_y1 = win_y + int(win_h * 0.12)
        opp2_y2 = win_y + int(win_h * 0.32)

        roi = img[opp2_y1:opp2_y2, opp2_x1:opp2_x2]
        if roi.size == 0:
            return 2

        pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        try:
            data = pytesseract.image_to_data(
                pil, config=self._tess_cfg_free, output_type=pytesseract.Output.DICT
            )
            for i, txt in enumerate(data["text"]):
                if len(txt.strip()) > 3 and int(data["conf"][i]) > 35:
                    return 3   # Texte trouvé → 2ème adversaire
        except Exception:
            pass
        return 2

    def _extract_meta(
        self, img: np.ndarray, state: PokerVisionState,
        win_x: int, win_y: int, win_w: int, win_h: int,
        scale: float = 1.0, board_y0: int = 0, card_h: int = REF_CARD_H
    ) -> None:
        """Extrait pot, stacks, blindes, actions, main_desc."""

        # ── Pot ────────────────────────────────────────────────────────
        pot_y1 = board_y0 + card_h + int(10 * scale)
        pot_y2 = board_y0 + card_h + int(70 * scale)
        pot_x1 = win_x + int(win_w * 0.38)
        pot_x2 = win_x + int(win_w * 0.65)
        state.pot = self._ocr_number_in(img, pot_y1, pot_y2, pot_x1, pot_x2,
                                        label="pot", pattern=r"(?:Pot\s*[:;]\s*)?(\d[\d\s]*)") or state.pot

        # ── Hero stack ─────────────────────────────────────────────────
        hero_st_y1 = board_y0 + int(450 * scale)
        hero_st_y2 = board_y0 + int(530 * scale)
        hero_st_x1 = win_x + int(win_w * 0.38)
        hero_st_x2 = win_x + int(win_w * 0.62)
        state.hero_stack = self._ocr_number_in(img, hero_st_y1, hero_st_y2,
                                               hero_st_x1, hero_st_x2, label="hero_stack") or state.hero_stack

        # ── Opponent stacks ────────────────────────────────────────────
        opp1_st_y1 = win_y + int(win_h * 0.20)
        opp1_st_y2 = win_y + int(win_h * 0.33)
        opp1_st_x1 = win_x + int(win_w * 0.16)
        opp1_st_x2 = win_x + int(win_w * 0.40)
        opp1_stack = self._ocr_number_in(img, opp1_st_y1, opp1_st_y2,
                                         opp1_st_x1, opp1_st_x2, label="opp1_stack")
        if opp1_stack:
            state.opp_stacks = [opp1_stack]

        # ── Blindes (coin supérieur droit) ─────────────────────────────
        blind_y1 = win_y + int(win_h * 0.03)
        blind_y2 = win_y + int(win_h * 0.12)
        blind_x1 = win_x + int(win_w * 0.78)
        blind_x2 = win_x + int(win_w * 0.96)
        state.blinds, state.level = self._ocr_blinds(img, blind_y1, blind_y2, blind_x1, blind_x2)

        # ── Actions dispos (boutons bas) ───────────────────────────────
        btn_y1 = win_y + int(win_h * 0.88)
        btn_y2 = win_y + int(win_h * 0.99)
        btn_x1 = win_x + int(win_w * 0.25)
        btn_x2 = win_x + int(win_w * 0.75)
        state.actions  = self._detect_actions(img, btn_y1, btn_y2, btn_x1, btn_x2)

        # ── Description main hero (ex : "2 Paires : 9-4") ─────────────
        desc_y1 = board_y0 + card_h + int(350 * scale)
        desc_y2 = board_y0 + card_h + int(430 * scale)
        desc_x1 = win_x + int(win_w * 0.30)
        desc_x2 = win_x + int(win_w * 0.70)
        state.hand_desc = self._ocr_text_in(img, desc_y1, desc_y2, desc_x1, desc_x2)

        # ── Position (BTN/SB/BB) ───────────────────────────────────────
        state.position = self._infer_position(img, win_x, win_y, win_w, win_h)

    # ── OCR helpers ─────────────────────────────────────────────────────────

    def _ocr_number_in(
        self, img: np.ndarray,
        y1: int, y2: int, x1: int, x2: int,
        label: str = "", pattern: str = r"(\d[\d\s]*)"
    ) -> Optional[float]:
        """OCR d'un nombre dans une zone, retourne float ou None."""
        roi = self._safe_crop(img, y1, y2, x1, x2)
        if roi is None:
            return None
        pil = self._enhance(roi)
        raw = pytesseract.image_to_string(pil, config=self._tess_cfg_num)
        m   = re.search(pattern, raw.replace(" ", ""))
        if m:
            try:
                return float(m.group(1).replace(" ", ""))
            except ValueError:
                pass
        return None

    def _ocr_blinds(
        self, img: np.ndarray,
        y1: int, y2: int, x1: int, x2: int
    ) -> Tuple[List[float], int]:
        """Parse '15-30' → ([15, 30], level)."""
        roi = self._safe_crop(img, y1, y2, x1, x2)
        if roi is None:
            return ([5.0, 10.0], 1)

        pil = self._enhance(roi, contrast=1.5)
        raw = pytesseract.image_to_string(pil, config="--psm 11 -c tessedit_char_whitelist=0123456789-.")
        m   = re.search(r"(\d+)[–\-](\d+)", raw)
        if m:
            sb = float(m.group(1))
            bb = float(m.group(2))
            level_m = re.search(r"Niv\w*\.?\s*(\d+)", raw, re.IGNORECASE)
            level = int(level_m.group(1)) if level_m else 1
            return ([sb, bb], level)
        return ([5.0, 10.0], 1)

    def _detect_actions(
        self, img: np.ndarray,
        y1: int, y2: int, x1: int, x2: int
    ) -> List[str]:
        """Détecte les boutons d'action disponibles (fold/check/call/raise)."""
        roi = self._safe_crop(img, y1, y2, x1, x2)
        if roi is None:
            return []

        pil = self._enhance(roi, contrast=1.8)
        raw = pytesseract.image_to_string(pil, config="--psm 11 -l fra+eng").lower()

        actions = []
        if "fold"  in raw: actions.append("fold")
        if "check" in raw: actions.append("check")
        if "call"  in raw: actions.append("call")
        if "raise" in raw or "mise" in raw: actions.append("raise")
        if "all"   in raw: actions.append("all_in")
        return actions

    def _ocr_text_in(
        self, img: np.ndarray,
        y1: int, y2: int, x1: int, x2: int
    ) -> str:
        """OCR texte libre dans une zone."""
        roi = self._safe_crop(img, y1, y2, x1, x2)
        if roi is None:
            return ""
        pil = self._enhance(roi)
        return pytesseract.image_to_string(pil, config="--psm 11 -l fra+eng").strip()

    def _infer_position(
        self, img: np.ndarray,
        win_x: int, win_y: int, win_w: int, win_h: int
    ) -> str:
        """
        Détecte la position (BTN/SB/BB) en cherchant le bouton 'D' (dealer).
        Le bouton D est généralement un disque jaune-orange.
        """
        # Zone centrale (table)
        roi = self._safe_crop(
            img,
            win_y + int(win_h * 0.30), win_y + int(win_h * 0.75),
            win_x + int(win_w * 0.15), win_x + int(win_w * 0.85)
        )
        if roi is None:
            return "unknown"

        # Cherche la couleur orange/jaune du bouton dealer
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (15, 100, 150), (35, 255, 255))  # orange-jaune
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if 100 < area < 5000:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                cx = bx + bw // 2
                cx_norm = cx / roi.shape[1]
                # BTN ≈ côté hero (centre-bas), SB/BB selon la position
                if cx_norm > 0.55:
                    return "BTN"
                elif cx_norm < 0.35:
                    return "BB"
                else:
                    return "SB"
        return "unknown"

    # ── Utilitaires ─────────────────────────────────────────────────────────

    def _safe_crop(
        self, img: np.ndarray,
        y1: int, y2: int, x1: int, x2: int
    ) -> Optional[np.ndarray]:
        """Crop sécurisé avec clipping aux limites de l'image."""
        h, w = img.shape[:2]
        y1 = max(0, min(y1, h - 1))
        y2 = max(y1 + 1, min(y2, h))
        x1 = max(0, min(x1, w - 1))
        x2 = max(x1 + 1, min(x2, w))
        crop = img[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    def _enhance(self, arr: np.ndarray, contrast: float = 2.0, scale: int = 3) -> Image.Image:
        """Agrandit + améliore contraste pour OCR."""
        big  = cv2.resize(arr, (arr.shape[1] * scale, arr.shape[0] * scale),
                          interpolation=cv2.INTER_LANCZOS4)
        pil  = Image.fromarray(cv2.cvtColor(big, cv2.COLOR_BGR2RGB))
        return ImageEnhance.Contrast(pil).enhance(contrast)


# ── Module-level convenience ─────────────────────────────────────────────────
_default_engine: Optional[ExpressoVision] = None


def extract_state(image_path_or_array, debug: bool = False) -> PokerVisionState:
    """
    Fonction utilitaire top-level.

    Usage :
        state = extract_state("/path/to/screenshot.png")
        game  = state.to_game_state()   # → dict pour route_request()
    """
    global _default_engine
    if _default_engine is None:
        _default_engine = ExpressoVision(debug=debug)

    if isinstance(image_path_or_array, str):
        return _default_engine.extract(image_path_or_array)
    else:
        return _default_engine.extract_from_array(image_path_or_array)


__all__ = ["ExpressoVision", "PokerVisionState", "CardInfo", "extract_state"]
