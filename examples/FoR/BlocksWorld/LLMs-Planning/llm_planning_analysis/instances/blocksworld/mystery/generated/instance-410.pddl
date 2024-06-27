(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects f i c a)
(:init 
(harmony)
(planet f)
(planet i)
(planet c)
(planet a)
(province f)
(province i)
(province c)
(province a)
)
(:goal
(and
(craves f i)
(craves i c)
(craves c a)
)))