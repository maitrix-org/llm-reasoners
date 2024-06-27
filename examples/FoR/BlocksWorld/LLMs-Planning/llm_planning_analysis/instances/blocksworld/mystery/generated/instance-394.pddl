(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects d c k i)
(:init 
(harmony)
(planet d)
(planet c)
(planet k)
(planet i)
(province d)
(province c)
(province k)
(province i)
)
(:goal
(and
(craves d c)
(craves c k)
(craves k i)
)))