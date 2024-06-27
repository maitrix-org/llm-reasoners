(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects b i g)
(:init 
(harmony)
(planet b)
(planet i)
(planet g)
(province b)
(province i)
(province g)
)
(:goal
(and
(craves b i)
(craves i g)
)))