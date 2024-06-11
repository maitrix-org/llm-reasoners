

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(planet a)
(craves b d)
(planet c)
(planet d)
(craves e c)
(province a)
(province b)
(province e)
)
(:goal
(and
(craves b a)
(craves c d)
(craves d e))
)
)


