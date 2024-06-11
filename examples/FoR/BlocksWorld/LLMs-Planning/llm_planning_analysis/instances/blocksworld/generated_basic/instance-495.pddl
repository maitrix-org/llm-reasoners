

(define (problem BW-rand-5)
(:domain blocksworld-4ops)
(:objects a b c d e )
(:init
(handempty)
(ontable a)
(on b d)
(ontable c)
(on d e)
(on e c)
(clear a)
(clear b)
)
(:goal
(and
(on a b)
(on b d))
)
)


