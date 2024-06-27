(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects h l e i)
(:init 
(harmony)
(planet h)
(planet l)
(planet e)
(planet i)
(province h)
(province l)
(province e)
(province i)
)
(:goal
(and
(craves h l)
(craves l e)
(craves e i)
)))