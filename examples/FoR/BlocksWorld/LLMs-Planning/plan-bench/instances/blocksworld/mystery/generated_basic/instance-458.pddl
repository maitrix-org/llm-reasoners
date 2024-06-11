

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(planet a)
(craves b c)
(craves c a)
(planet d)
(planet e)
(province b)
(province d)
(province e)
)
(:goal
(and
(craves a c)
(craves b e)
(craves c b)
(craves e d))
)
)


