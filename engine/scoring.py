"""Duplicate and rubber bridge contract scoring.

This module implements the Laws of Duplicate Contract Bridge scoring
(and a rubber bridge variant).  All scores are expressed from the
*declarer's* point of view:

- A **positive** return value means declarer's side earns that many points
  (making the contract).
- A **negative** return value means declarer's side concedes that many points
  to the defenders (going down).

Typical usage::

    from .card import Suit, Contract
    pts = score(contract, declarer=0, doubled=0, tricks_made=10, vulnerable=False)
"""

from .card import Suit

def score(contract, declarer, doubled, tricks_made, vulnerable):
    """Score a completed duplicate bridge contract.

    Args:
        contract: A Contract object with ``level`` (1-7) and ``strain``
            (Suit.C / Suit.D / Suit.H / Suit.S / Suit.NT) attributes.
        declarer: Seat index of the declarer (0-3).  Not used in the
            calculation itself but kept for call-site symmetry with
            ``score_rubber``.
        doubled: Doubling status of the contract.
            0 = undoubled, 1 = doubled, 2 = redoubled.
        tricks_made: Total tricks won by declarer (0-13).
        vulnerable: ``True`` if declarer's side is vulnerable,
            ``False`` otherwise.

    Returns:
        int: Net score for declarer's side.  Positive when the contract
        is made (trick score + game/slam bonuses + overtricks +
        insult bonus).  Negative when the contract is defeated
        (undertrick penalties expressed as a negative number).
    """
    level = contract.level
    strain = contract.strain
    target = 6 + level
    overtricks = tricks_made - target
    undertricks = target - tricks_made

    if tricks_made < target:
        return _undertrick_penalty(undertricks, doubled, vulnerable)

    trick_score = _trick_score(level, strain, doubled)
    game = trick_score >= 100
    bonus = _game_bonus(game, level, vulnerable) + _slam_bonus(level, vulnerable)
    overtrick_score = _overtrick_score(overtricks, strain, doubled, vulnerable)
    double_bonus = 50 if doubled == 1 else 100 if doubled == 2 else 0

    total = trick_score + bonus + overtrick_score + double_bonus
    print(f"[SCORE] Contract:{contract} Doubled:{doubled} Tricks:{tricks_made}/{target} Vul:{vulnerable}")
    print(f"  Trick score:{trick_score} Bonus:{bonus} Overtricks:{overtrick_score} DblBonus:{double_bonus} => {total}")
    return total

def _trick_score(level, strain, doubled):
    """Calculate the raw trick score for a made contract.

    The trick score is the *below-the-line* value that counts toward game:

    - Minor suits (clubs / diamonds): 20 points per trick bid.
    - Major suits (hearts / spades): 30 points per trick bid.
    - No-trump: 40 for the first trick bid (30 + 10 NT premium), then
      30 for each subsequent trick bid.

    Doubled contracts multiply the base score by 2; redoubled contracts
    multiply by 4.

    Args:
        level: Bid level (1-7).
        strain: The trump strain (Suit.C / Suit.D / Suit.H / Suit.S / Suit.NT).
        doubled: Doubling status — 0 undoubled, 1 doubled, 2 redoubled.

    Returns:
        int: Trick score before bonuses.
    """
    if strain in (Suit.C, Suit.D):
        per_trick = 20
    elif strain in (Suit.H, Suit.S):
        per_trick = 30
    else:
        per_trick = 30
    first_nt = 10 if strain == Suit.NT else 0
    base = first_nt + per_trick * level
    if doubled == 1: return base * 2
    if doubled == 2: return base * 4
    return base

def _game_bonus(game, level, vulnerable):
    """Return the part-score or game bonus for a made contract.

    A contract is a *game* when its trick score (before doubling
    adjustments that affect the base trick score) is 100 or more.
    The threshold is evaluated on the already-multiplied trick score
    passed in via ``game``.

    Args:
        game: ``True`` if the trick score reaches 100 (game-level contract).
        level: Bid level (1-7).  Unused here but kept for a consistent
            helper signature.
        vulnerable: ``True`` if declarer's side is vulnerable.

    Returns:
        int: 500 for a vulnerable game, 300 for a non-vulnerable game,
        or 50 for a part-score.
    """
    if game: return 500 if vulnerable else 300
    return 50

def _slam_bonus(level, vulnerable):
    """Return the slam bonus for a made small or grand slam.

    Slam bonuses are awarded *in addition* to the game bonus:

    - Small slam (level 6): 750 vulnerable, 500 not vulnerable.
    - Grand slam (level 7): 1500 vulnerable, 1000 not vulnerable.
    - All other levels: 0.

    Args:
        level: Bid level (1-7).
        vulnerable: ``True`` if declarer's side is vulnerable.

    Returns:
        int: Slam bonus points, or 0 if no slam was bid.
    """
    if level == 7: return 1500 if vulnerable else 1000
    if level == 6: return 750 if vulnerable else 500
    return 0

def _overtrick_score(overtricks, strain, doubled, vulnerable):
    """Calculate the score for tricks taken beyond the contract.

    Overtrick scoring depends on whether the contract was doubled:

    - Undoubled: same per-trick rate as the contract strain
      (20 for minors, 30 for majors / NT).
    - Doubled: 100 per overtrick not vulnerable, 200 vulnerable.
    - Redoubled: 200 per overtrick not vulnerable, 400 vulnerable.

    Args:
        overtricks: Number of tricks made above the contract target.
            0 or negative values return 0.
        strain: The trump strain (Suit.C / Suit.D / Suit.H / Suit.S / Suit.NT).
        doubled: Doubling status — 0 undoubled, 1 doubled, 2 redoubled.
        vulnerable: ``True`` if declarer's side is vulnerable.

    Returns:
        int: Total overtrick score (always >= 0).
    """
    if overtricks <= 0: return 0
    if doubled == 0:
        per = 20 if strain in (Suit.C, Suit.D) else 30
        return per * overtricks
    if doubled == 1:
        per = 200 if vulnerable else 100
        return per * overtricks
    per = 400 if vulnerable else 200
    return per * overtricks

def _undertrick_penalty(undertricks, doubled, vulnerable):
    """Calculate the penalty for a defeated contract.

    Penalty scoring (from declarer's perspective, so always <= 0):

    - Undoubled: 50 per trick not vulnerable, 100 per trick vulnerable.
    - Doubled: see ``_doubled_penalty`` (non-linear scale).
    - Redoubled: double the doubled penalty.

    Args:
        undertricks: Number of tricks by which the contract was defeated
            (always >= 1 when this helper is called).
        doubled: Doubling status — 0 undoubled, 1 doubled, 2 redoubled.
        vulnerable: ``True`` if declarer's side is vulnerable.

    Returns:
        int: Penalty as a **negative** integer (e.g. -100 for one down
        undoubled not vulnerable).
    """
    if doubled == 0:
        per = 100 if vulnerable else 50
        return -(per * undertricks)
    if doubled == 1:
        return -_doubled_penalty(undertricks, vulnerable)
    return -_doubled_penalty(undertricks, vulnerable) * 2

def _doubled_penalty(n, vulnerable):
    """Return the raw (positive) doubled undertrick penalty.

    Vulnerable scale: 200 for the first undertrick, then 300 for each
    additional undertrick (200 + 300*(n-1)).

    Not-vulnerable scale (non-linear):
    - 1 down: 100
    - 2 down: 300  (100 + 200)
    - 3 down: 500  (100 + 200 + 200)
    - 4+ down: 500 + 300 * (n - 3)

    This value is negated by the caller (``_undertrick_penalty``) to
    produce a negative score for declarer, and doubled again for
    redoubled contracts.

    Args:
        n: Number of undertricks (>= 1).
        vulnerable: ``True`` if declarer's side is vulnerable.

    Returns:
        int: Positive penalty amount before sign inversion.
    """
    if vulnerable:
        return 200 + 300 * (n - 1)
    else:
        if n == 1: return 100
        if n == 2: return 300
        if n == 3: return 500
        return 500 + 300 * (n - 3)

def score_rubber(contract, declarer, doubled, tricks_made, vulnerable, below_line_ns, below_line_ew):
    """Score a completed rubber bridge contract, splitting below/above the line.

    In rubber bridge, the trick score goes *below the line* (counting toward
    winning a game) while all bonuses and penalties go *above the line*.
    This function updates the running below-the-line totals and returns the
    above-the-line deltas for both sides.

    Slam bonuses and rubber/game bonuses are **not** computed here; they are
    awarded when a rubber ends (outside this function).

    Args:
        contract: A Contract object with ``level`` (1-7) and ``strain``
            (Suit.C / Suit.D / Suit.H / Suit.S / Suit.NT) attributes.
        declarer: Seat index of the declarer (0-3).  Even indices (0, 2)
            are North-South; odd indices (1, 3) are East-West.
        doubled: Doubling status — 0 undoubled, 1 doubled, 2 redoubled.
        tricks_made: Total tricks won by declarer (0-13).
        vulnerable: ``True`` if declarer's side is vulnerable.
        below_line_ns: Current below-the-line total for North-South before
            this hand.
        below_line_ew: Current below-the-line total for East-West before
            this hand.

    Returns:
        tuple[int, int, int, int]: A 4-tuple of
        ``(above_ns, above_ew, new_below_ns, new_below_ew)`` where:

        - ``above_ns``: Points scored above the line by North-South this
          hand (overtricks, insult bonus, or penalty received).
        - ``above_ew``: Points scored above the line by East-West this
          hand.
        - ``new_below_ns``: Updated below-the-line total for North-South.
        - ``new_below_ew``: Updated below-the-line total for East-West.

        If the contract is defeated, the penalty is credited above the line
        to the non-declaring side and the below-the-line totals are
        unchanged.
    """
    level = contract.level
    strain = contract.strain
    target = 6 + level
    overtricks = tricks_made - target
    undertricks = target - tricks_made
    is_dec_ns = declarer % 2 == 0

    if tricks_made < target:
        penalty = -_undertrick_penalty(undertricks, doubled, vulnerable)
        if is_dec_ns: return 0, penalty, below_line_ns, below_line_ew
        else: return penalty, 0, below_line_ns, below_line_ew

    trick_score = _trick_score(level, strain, doubled)
    overtime = _overtrick_score(overtricks, strain, doubled, vulnerable)
    double_bonus = 50 if doubled == 1 else 100 if doubled == 2 else 0

    new_bns = below_line_ns + (trick_score if is_dec_ns else 0)
    new_bew = below_line_ew + (trick_score if not is_dec_ns else 0)
    above = overtime + double_bonus

    return (above, 0, new_bns, new_bew) if is_dec_ns else (0, above, new_bns, new_bew)
