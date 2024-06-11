

(define (problem BW-rand-5)
(:domain blocksworld-4ops)
(:objects a b c d e )
(:init
(handempty)
(on a e)
(on b d)
(on c b)
(ontable d)
(on e c)
(clear a)
)
(:goal
(and
(on c d)
(on d e))
)
)


