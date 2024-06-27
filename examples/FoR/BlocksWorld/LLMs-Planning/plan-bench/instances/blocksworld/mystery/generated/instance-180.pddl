(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects f c i l)
(:init 
(harmony)
(planet f)
(planet c)
(planet i)
(planet l)
(province f)
(province c)
(province i)
(province l)
)
(:goal
(and
(craves f c)
(craves c i)
(craves i l)
)))