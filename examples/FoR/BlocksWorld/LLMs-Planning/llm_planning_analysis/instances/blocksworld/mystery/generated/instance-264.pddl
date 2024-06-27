(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects f e i c)
(:init 
(harmony)
(planet f)
(planet e)
(planet i)
(planet c)
(province f)
(province e)
(province i)
(province c)
)
(:goal
(and
(craves f e)
(craves e i)
(craves i c)
)))