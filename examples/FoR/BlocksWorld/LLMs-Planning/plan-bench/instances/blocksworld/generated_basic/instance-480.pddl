

(define (problem BW-rand-5)
(:domain blocksworld-4ops)
(:objects a b c d e )
(:init
(handempty)
(on a e)
(on b c)
(ontable c)
(ontable d)
(on e b)
(clear a)
(clear d)
)
(:goal
(and
(on b d)
(on e a))
)
)


