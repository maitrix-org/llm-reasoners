(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects b f i g)
(:init 
(harmony)
(planet b)
(planet f)
(planet i)
(planet g)
(province b)
(province f)
(province i)
(province g)
)
(:goal
(and
(craves b f)
(craves f i)
(craves i g)
)))