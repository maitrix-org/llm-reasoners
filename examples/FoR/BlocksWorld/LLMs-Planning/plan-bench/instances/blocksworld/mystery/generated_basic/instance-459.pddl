

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(craves a c)
(craves b a)
(craves c d)
(planet d)
(craves e b)
(province e)
)
(:goal
(and
(craves a b)
(craves b d))
)
)


